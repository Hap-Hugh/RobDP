//
// Created by Xuan Chen on 2025/9/11.
//

#include "optimizer/error_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "c.h"

/* Pair for sorting distances while keeping original index. */
typedef struct {
    double d;
    int i;
} Pair;

/* Clamp a double into [0, 1]. */
static double clamp01(double x) {
    if (x < 0) return 0;
    if (x > 1) return 1;
    return x;
}

/* Max/Min helpers for double. */
static double maxd(double a, double b) { return a > b ? a : b; }
static double mind(double a, double b) { return a < b ? a : b; }

/*
 * Compute mean and (unbiased) standard deviation.
 * If n == 0 or n == 1, std is set to 0.
 */
static void mean_std(const double *x, int n, double *mean, double *std) {
    if (n <= 0) {
        if (mean) *mean = 0;
        if (std) *std = 0;
        return;
    }
    double m = 0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= (double) n;

    double v = 0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - m;
        v += d * d;
    }
    v = (n > 1) ? v / (double) (n - 1) : 0.0;

    if (mean) *mean = m;
    if (std) *std = sqrt(v);
}

/*
 * Silverman's rule-of-thumb bandwidth for 1D KDE:
 *   h = 1.06 * std * n^(-1/5)
 * Falls back to a small positive value if std <= 0 or n < 2.
 */
static double silverman_bandwidth(double std, int n) {
    if (n < 2 || std <= 0) return 1e-3;
    return 1.06 * std * pow((double) n, -0.2);
}

/* qsort comparator for doubles (ascending). */
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *) a, y = *(const double *) b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/*
 * Estimate IQR (Q3 - Q1) using simple index-based quartiles after sorting.
 * The input array 'tmp' is sorted in-place.
 */
static double estimate_iqr(double *tmp, int n) {
    if (n <= 0) return 0.0;
    qsort(tmp, n, sizeof(double), cmp_double);
    int q1i = (int) (0.25 * (n - 1));
    int q3i = (int) (0.75 * (n - 1));
    return tmp[q3i] - tmp[q1i];
}

/*
 * Gaussian kernel (unnormalized): K(u) = exp(-0.5*u^2).
 * Normalization constant is not needed for relative weighting / sampling.
 */
static inline double gaussian_kernel_u(double u) {
    return exp(-0.5 * u * u);
}

/*
 * Reentrant normal(0,1) via Box–Muller using rand_r().
 * Note: rand_r() is POSIX and not C11; consider replacing in PostgreSQL code.
 */
static double randn(unsigned int *state) {
    double u1 = (rand_r(state) + 1.0) / ((double) RAND_MAX + 2.0);
    double u2 = (rand_r(state) + 1.0) / ((double) RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/*
 * Load an error profile from "<base_dir>/<alias>.txt".
 * Each line is: <true_selectivity> <estimated_selectivity>
 * Returns 0 on success. On failure, returns a nonzero code and leaves *out zeroed.
 *
 * Populates:
 *  - out->data: array of EPSample { true_sel, est_sel } (malloc'ed)
 *  - out->n:    number of samples
 *  - out->est_std / out->true_std: global stddevs for est/true components
 */
int load_error_profile(const char *base_dir, const char *alias, ErrorProfile *out) {
    if (!base_dir || !alias || !out) return 1;
    MemSet(out, 0, sizeof(*out));

    /* Build path like "/opt/17a/<alias>.txt". */
    char path[4096];
    MemSet(path, 0, sizeof(path));
    snprintf(path, sizeof(path), "%s/%s.txt", base_dir, alias);

    FILE *fp = fopen(path, "r");
    if (!fp) return 2;

    /* Preallocate buffer and grow geometrically as needed. */
    int cap = 1 << 16; /* 65536 */
    EPSample *buf = (EPSample *) malloc(cap * sizeof(EPSample));
    if (!buf) {
        fclose(fp);
        return 3;
    }

    int n = 0;
    for (;;) {
        double t, e;
        int r = fscanf(fp, "%lf %lf", &t, &e);
        if (r == EOF) break; /* end of file */
        if (r != 2) {
            /* Skip malformed line: discard until EOL. */
            int c;
            while ((c = fgetc(fp)) != EOF && c != '\n') {
                /* no-op */
            }
            continue;
        }
        if (n == cap) {
            cap <<= 1;
            EPSample *nbuf = (EPSample *) realloc(buf, cap * sizeof(EPSample));
            if (!nbuf) {
                free(buf);
                fclose(fp);
                return 4;
            }
            buf = nbuf;
        }
        buf[n].true_sel = clamp01(t);
        buf[n].est_sel = clamp01(e);
        n++;
    }
    fclose(fp);

    if (n == 0) {
        free(buf);
        return 5;
    }

    out->data = buf;
    out->n = n;

    /* Compute stddevs for estimated and true selectivities. */
    double *ests = (double *) malloc(n * sizeof(double));
    double *trus = (double *) malloc(n * sizeof(double));
    if (!ests || !trus) {
        free(ests);
        free(trus);
        free(buf);
        MemSet(out, 0, sizeof(*out));
        return 6;
    }
    for (int i = 0; i < n; i++) {
        ests[i] = buf[i].est_sel;
        trus[i] = buf[i].true_sel;
    }
    double dummy_mean, s_est, s_true;
    mean_std(ests, n, &dummy_mean, &s_est);
    mean_std(trus, n, &dummy_mean, &s_true);
    out->est_std = s_est;
    out->true_std = s_true;

    free(ests);
    free(trus);
    return 0;
}

/* Free storage inside an ErrorProfile (safe on partially-filled structs). */
void free_error_profile(ErrorProfile *ep) {
    if (!ep) return;
    free(ep->data);
    ep->data = NULL;
    ep->n = 0;
    ep->est_std = 0;
    ep->true_std = 0;
}

/* qsort comparator for Pair by ascending distance. */
static int cmp(const void *a, const void *b) {
    double x = ((const Pair *) a)->d, y = ((const Pair *) b)->d;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/*
 * Return indices of the k nearest neighbors in 'ep' w.r.t. |est_sel - e0|.
 * Writes into out_idx (size >= k). Returns the number of indices written (<= k).
 * Complexity: O(n log n) due to full sort; good enough for moderate n.
 */
static int knn_indices_by_est(const ErrorProfile *ep, double e0, int k, int *out_idx) {
    int n = ep->n;
    Pair *arr = (Pair *) malloc(n * sizeof(Pair));
    if (!arr) return 0;

    for (int i = 0; i < n; i++) {
        arr[i].d = fabs(ep->data[i].est_sel - e0);
        arr[i].i = i;
    }
    qsort(arr, n, sizeof(Pair), cmp);

    int m = (k < n) ? k : n;
    for (int j = 0; j < m; j++) out_idx[j] = arr[j].i;

    free(arr);
    return m;
}

Distribution *build_conditional_distribution(
    const ErrorProfile *ep,
    double e0,
    int n_samples,
    double h_est,
    double h_true, /* now used as log-error bandwidth */
    unsigned int seed
) {
    if (!ep || ep->n == 0 || n_samples <= 0) return NULL;

    if (seed == 0) seed = (unsigned int) time(NULL);

    /* ---------- 1) KDE bandwidth in est_sel axis (unchanged) ---------- */
    if (h_est <= 0) {
        double *tmp = (double *) malloc(ep->n * sizeof(double));
        if (!tmp) return NULL;
        for (int i = 0; i < ep->n; i++) tmp[i] = ep->data[i].est_sel;

        double mean, std;
        mean_std(tmp, ep->n, &mean, &std);
        double iqr = estimate_iqr(tmp, ep->n);
        free(tmp);

        double robust_sigma = (iqr > 0) ? mind(std, iqr / 1.349) : std;
        h_est = silverman_bandwidth(robust_sigma, ep->n);
        h_est = maxd(h_est, 1e-3);
    }

    /* ---------- 2) Precompute log-errors: eps_i = log(T_i) - log(E_i) ---------- */
    const double EPS = 1e-15; /* avoid log(0). Tune if needed. */
    int n = ep->n;
    double *errs = (double *) malloc(n * sizeof(double));
    if (!errs) return NULL;

    for (int i = 0; i < n; i++) {
        double T = ep->data[i].true_sel;
        double E = ep->data[i].est_sel;
        /* clamp into (0,1]; keep 1 as-is, floor near 0 to EPS */
        double Tc = (T <= 0.0) ? EPS : T;
        double Ec = (E <= 0.0) ? EPS : E;
        errs[i] = log(Tc) - log(Ec);
    }

    /* ---------- 3) Bandwidth in log-error axis (if not provided) ---------- */
    if (h_true <= 0) {
        /* Use std/IQR of errs for a robust bandwidth in log-error space */
        double mean, std;
        mean_std(errs, n, &mean, &std);

        /* Build a scratch copy for IQR (estimate_iqr sorts in-place) */
        double *tmp2 = (double *) malloc(n * sizeof(double));
        if (!tmp2) {
            free(errs);
            return NULL;
        }
        for (int i = 0; i < n; i++) tmp2[i] = errs[i];

        double iqr = estimate_iqr(tmp2, n);
        free(tmp2);

        double robust_sigma = (iqr > 0) ? mind(std, iqr / 1.349) : std;
        /* Smaller jitter than KDE bandwidth; feel free to tune factor (e.g., 0.5) */
        h_true = 0.25 * silverman_bandwidth(robust_sigma, n);
        h_true = maxd(h_true, 1e-3);
    }

    /* ---------- 4) Kernel weights in est_sel axis (unchanged) ---------- */
    double *w = (double *) malloc(n * sizeof(double));
    if (!w) {
        free(errs);
        return NULL;
    }

    double sumw = 0.0, inv_h = 1.0 / h_est;
    for (int i = 0; i < n; i++) {
        double u = (ep->data[i].est_sel - e0) * inv_h;
        double wi = gaussian_kernel_u(u);
        w[i] = wi;
        sumw += wi;
    }

    /* KNN fallback if all weights ~ 0 (unchanged logic) */
    if (sumw < 1e-12) {
        int k = (int) fmin((double) n, fmax(50.0, sqrt((double) n)));
        int *idx = (int *) malloc(k * sizeof(int));
        if (!idx) {
            free(w);
            free(errs);
            return NULL;
        }
        int m = knn_indices_by_est(ep, e0, k, idx);

        for (int i = 0; i < n; i++) w[i] = 0.0;
        for (int j = 0; j < m; j++) w[idx[j]] = 1.0;
        sumw = (double) m;

        free(idx);
    }

    /* ---------- 5) Build CDF for inverse-CDF sampling (unchanged) ---------- */
    double *cdf = (double *) malloc(n * sizeof(double));
    if (!cdf) {
        free(w);
        free(errs);
        return NULL;
    }

    double acc = 0.0;
    if (sumw > 0) {
        for (int i = 0; i < n; i++) {
            acc += w[i] / sumw;
            cdf[i] = acc;
        }
        cdf[n - 1] = 1.0;
    } else {
        for (int i = 0; i < n; i++) cdf[i] = (i + 1) / (double) n;
    }

    /* ---------- 6) Allocate output Distribution ---------- */
    Distribution *dist = (Distribution *) malloc(sizeof(Distribution));
    if (!dist) {
        free(cdf);
        free(w);
        free(errs);
        return NULL;
    }

    dist->sample_count = n_samples;
    dist->probs = (double *) malloc(n_samples * sizeof(double));
    dist->vals = (double *) malloc(n_samples * sizeof(double));
    if (!dist->probs || !dist->vals) {
        free(dist->probs);
        free(dist->vals);
        free(dist);
        free(cdf);
        free(w);
        free(errs);
        return NULL;
    }

    /* ---------- 7) Sample err ~ p(err | est≈e0), then map to true_sel via exp ---------- */
    for (int j = 0; j < n_samples; j++) {
        double u = (rand_r(&seed) + 1.0) / ((double) RAND_MAX + 2.0);

        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (cdf[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        int pick = lo;

        double err_draw = errs[pick] + h_true * randn(&seed); /* log-error jitter */
        double t_draw = e0 * exp(err_draw);

        /* Clamp to [0,1]; e0=0 degenerates to 0 as desired. */
        dist->vals[j] = clamp01(t_draw);
        dist->probs[j] = 1.0 / (double) n_samples;
    }

    free(cdf);
    free(w);
    free(errs);
    return dist;
}

Distribution *scale_distribution(const Distribution *src, double factor) {
    if (!src) {
        return NULL;
    }
    Distribution *dst = malloc(sizeof(Distribution));
    if (!dst) {
        return NULL;
    }
    dst->sample_count = src->sample_count;
    dst->probs = malloc(dst->sample_count * sizeof(double));
    dst->vals = malloc(dst->sample_count * sizeof(double));

    if (!dst->probs || !dst->vals) {
        free(dst->probs);
        free(dst->vals);
        free(dst);
        return NULL;
    }
    for (int i = 0; i < dst->sample_count; i++) {
        dst->probs[i] = src->probs[i];
        dst->vals[i] = src->vals[i] * factor;
    }
    return dst;
}

/* Free storage inside a Distribution. */
void free_distribution(Distribution *dist) {
    if (!dist) return;
    free(dist->probs);
    free(dist->vals);
    free(dist);
}
