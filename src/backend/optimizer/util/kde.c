//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/10/2.
// Modified by Xuan Chen on 2025/10/5.
//

#include "postgres.h"

#include <math.h>
#include <stdlib.h>
#include "optimizer/sample.h"
#include "optimizer/ep.h"
#include "optimizer/kde.h"

static const double EPS = 1e-12;

/* GUC Parameters */
double error_sample_kde_bandwidth = 0.5;

/* ---------------- Utilities (KNN)---------------- */
/* Pair for sorting distances while keeping original index. */
typedef struct {
    double d;
    int i;
} Pair;

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
    int n = ep->sample_count;
    Pair *arr = (Pair *) palloc0(n * sizeof(Pair));
    if (!arr) return 0;

    for (int i = 0; i < n; i++) {
        arr[i].d = fabs(ep->data[i].sel_est - e0);
        arr[i].i = i;
    }
    qsort(arr, n, sizeof(Pair), cmp);

    int m = (k < n) ? k : n;
    for (int j = 0; j < m; j++) out_idx[j] = arr[j].i;

    pfree(arr);
    return m;
}

/* ---------------- Utilities (KDE)---------------- */
/* Max/Min helpers for double. */
static double maxd(double a, double b) {
    return a > b ? a : b;
}

static double mind(double a, double b) {
    return a < b ? a : b;
}

/*
 * Compute mean and (unbiased) standard deviation.
 */
static void mean_std(const double *x, int n, double *mean, double *std) {
    double m = 0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= (double) n;

    double v = 0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - m;
        v += d * d;
    }
    v = (n > 1) ? v / (double) (n - 1) : 0.0;

    *mean = m;
    *std = sqrt(v);
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

static uint32_t xs32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = (x == 0u) ? 0x6d2b79f5u : x; /* avoid zero state */
    return *s;
}

static double urand01(uint32_t *s) {
    /* 24-bit mantissa uniform in [0,1) */
    return (xs32(s) >> 8) * (1.0 / 16777216.0);
}

static double randn_box_muller(uint32_t *s) {
    /* One standard normal via Box–Muller (polar form could also be used) */
    double u1, u2;
    do {
        u1 = urand01(s);
    } while (u1 <= 0.0);
    u2 = urand01(s);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
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
    qsort(tmp, n, sizeof(double), cmp_double);
    int q1i = (int) (0.25 * (n - 1));
    int q3i = (int) (0.75 * (n - 1));
    return tmp[q3i] - tmp[q1i];
}

/*
 * Gaussian kernel (unnormalized): K(u) = exp(-0.5*u^2).
 * Normalization constant is not needed for relative weighting / sampling.
 */
static double gaussian_kernel_u(double u) {
    return exp(-0.5 * u * u);
}

/* ------------------------------- KDE Sampling ------------------------------- */
Sample *make_sample_by_condition(const ErrorProfile *ep, double sel_est) {
    /* ---------- 0) Basic validation ---------- */
    if (!ep) return NULL;
    int n = ep->sample_count;
    if (n <= 0) return NULL;

    int n_samples = error_sample_count;
    if (n_samples <= 0 || n_samples > DIST_MAX_SAMPLE)
        return NULL;

    /* Seed setup */
    uint32_t rng = error_sample_seed;

    /* ---------- 1) KDE bandwidth on est_sel axis (Silverman, robust) ---------- */
    double h_est = 0.0;
    {
        /* If you want a fixed bandwidth, set h_est directly here. */
        double *tmp = (double *) palloc0(n * sizeof(double));
        if (!tmp) return NULL;
        for (int i = 0; i < n; i++)
            tmp[i] = ep->data[i].sel_est;

        double mean, std;
        mean_std(tmp, n, &mean, &std);

        /* estimate_iqr may sort; use it on a scratch copy if needed */
        double iqr = estimate_iqr(tmp, n);

        double robust_sigma = (iqr > 0.0) ? mind(std, iqr / 1.349) : std;
        h_est = silverman_bandwidth(robust_sigma, n);
        h_est = maxd(h_est, 1e-3); /* lower bound to avoid divide-by-zero */

        pfree(tmp);
    }

    /* ---------- 2) Precompute log-errors: err_i = log(T_i) - log(E_i) ---------- */
    double *errs = (double *) palloc0(n * sizeof(double));
    if (!errs) return NULL;

    for (int i = 0; i < n; i++) {
        double T = ep->data[i].sel_true;
        double E = ep->data[i].sel_est;
        /* Clamp into (0,1]; floor near 0 to EPS to avoid log(0). */
        double Tc = (T <= 0.0) ? EPS : T;
        double Ec = (E <= 0.0) ? EPS : E;
        errs[i] = log(Tc) - log(Ec);
    }

    /* ---------- 3) Bandwidth in log-error axis (GUC or robust auto) ---------- */
    double h_err = error_sample_kde_bandwidth; /* if >0, use it as-is */
    if (h_err <= 0.0) {
        double mean, std;
        mean_std(errs, n, &mean, &std);

        /* build a scratch copy for IQR (estimate_iqr may reorder) */
        double *tmp2 = (double *) palloc0(n * sizeof(double));
        if (!tmp2) {
            pfree(errs);
            return NULL;
        }
        for (int i = 0; i < n; i++) tmp2[i] = errs[i];

        double iqr = estimate_iqr(tmp2, n);
        pfree(tmp2);

        double robust_sigma = (iqr > 0.0) ? mind(std, iqr / 1.349) : std;

        /* Slightly tighter than Silverman on raw sigma (tune factor as needed) */
        h_err = 0.25 * silverman_bandwidth(robust_sigma, n);
        h_err = maxd(h_err, 1e-3);
    }

    /* ---------- 4) Kernel weights in est_sel axis around e0 ---------- */
    double *w = (double *) palloc0(n * sizeof(double));
    if (!w) {
        pfree(errs);
        return NULL;
    }

    double inv_h_est = 1.0 / h_est;
    double sumw = 0.0;

    for (int i = 0; i < n; i++) {
        double u = (ep->data[i].sel_est - sel_est) * inv_h_est;
        double wi = gaussian_kernel_u(u); /* e.g., exp(-0.5*u^2) / sqrt(2π) or unnormalized */
        w[i] = wi;
        sumw += wi;
    }

    /* ---------- 5) KNN fallback if weights ~ 0 ---------- */
    if (sumw < 1e-12) {
        int k = (int) fmin((double) n, fmax(50.0, sqrt((double) n)));
        int *idx = (int *) palloc0(k * sizeof(int));
        if (!idx) {
            pfree(w);
            pfree(errs);
            return NULL;
        }

        int m = knn_indices_by_est(ep, sel_est, k, idx); /* returns <= k indices */

        for (int i = 0; i < n; i++)
            w[i] = 0.0;
        for (int j = 0; j < m; j++)
            w[idx[j]] = 1.0;

        sumw = (double) m;
        pfree(idx);

        /* If still zero (empty ep?), abort gracefully */
        if (sumw <= 0.0) {
            pfree(w);
            pfree(errs);
            return NULL;
        }
    }

    /* ---------- 6) Normalize weights + build CDF for mixture sampling ---------- */
    double *cdf = (double *) palloc0(n * sizeof(double));
    if (!cdf) {
        pfree(w);
        pfree(errs);
        return NULL;
    }

    double acc = 0.0, inv_sumw = 1.0 / sumw;
    for (int i = 0; i < n; i++) {
        acc += w[i] * inv_sumw;
        cdf[i] = acc;
    }
    /* Ensure final CDF ends at 1.0 numerically */
    cdf[n - 1] = 1.0;

    /* ---------- 7) Draw samples from the log-error mixture and map back ---------- */
    Sample *S = (Sample *) palloc0(sizeof(Sample));
    if (!S) {
        pfree(cdf);
        pfree(w);
        pfree(errs);
        return NULL;
    }

    S->sample_count = n_samples;

    for (int s = 0; s < n_samples; s++) {
        /* (a) Choose a component i ~ Categorical(w) using inverse-CDF */
        double u = urand01(&rng);

        int lo = 0, hi = n - 1, pick = n - 1;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (cdf[mid] >= u) {
                pick = mid;
                hi = mid - 1;
            } else { lo = mid + 1; }
        }

        /* (b) Draw log-error from N(errs[pick], h_err^2) */
        double z = randn_box_muller(&rng);
        double log_err = errs[pick] + h_err * z;

        /* (c) Map back: T = E0 * exp(log_err); clamp into (EPS,1] */
        double T = sel_est * exp(log_err);
        if (T <= EPS) T = EPS;
        if (T > 1.0) T = 1.0;

        S->sample[s] = T;
    }

    /* ---------- 8) Cleanup scratch ---------- */
    pfree(cdf);
    pfree(w);
    pfree(errs);

    return S;
}
