/*-------------------------------------------------------------------------
 *
 * kde_conditional.c
 *    Build a conditional distribution p(true_sel | est_sel ≈ e0) from an
 *    error profile using a 1D KDE over the estimate axis and a log-error
 *    model for true vs. estimated selectivity.
 *
 *    Created: 2025-09-22
 *
 *    This file depends on:
 *      - postgres.h (palloc/pfree, ereport, etc.)
 *      - optimizer/dist.h    (Distribution)
 *      - optimizer/kde.h     (ErrorProfile, calc_mean_std, clamp01, etc.)
 *
 *    High-level:
 *      1) Choose a bandwidth h_est on the estimate axis (Silverman/robust).
 *      2) Pre-compute per-sample log-errors: err_i = log(T_i) - log(E_i).
 *      3) Choose a bandwidth h_true for jitter in log-error space.
 *      4) Compute kernel weights w_i = K((E_i - e0)/h_est).
 *         - If all weights ~0, fall back to a KNN mask near e0.
 *      5) Build a CDF over the (possibly reweighted) samples.
 *      6) Sample indices via inverse-CDF; for each, draw a log-error by
 *         adding N(0, h_true^2); map back: T_draw = e0 * exp(err_draw).
 *      7) Clamp to [0, 1] and return an empirical distribution.
 *
 *    Notes:
 *      - Kernels are unnormalized since we only need relative weights.
 *      - Bandwidths use a robust sigma = min(std, IQR/1.349).
 *      - Randomness uses rand_r for reentrancy; see randn()/uniform01().
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include <math.h>
#include <stdlib.h>
#include "optimizer/dist.h"
#include "optimizer/kde.h"

static const double EPS = 1e-12;
double error_kde_sigma_span = 4.0;

/* GUC Parameters */
int error_bin_count = 1;
double error_sample_kde_bandwidth = 0.0;

/* ---------------- Utilities ---------------- */

/* Global/static variable to hold context for comparator */
static const Distribution *g_dist_for_sort = NULL;

/* Standard qsort comparator: no ctx argument */
static int cmp_index_desc(const void *a, const void *b) {
    int ia = *(const int *) a;
    int ib = *(const int *) b;

    if (g_dist_for_sort->probs[ia] < g_dist_for_sort->probs[ib])
        return 1; /* larger prob first */
    else if (g_dist_for_sort->probs[ia] > g_dist_for_sort->probs[ib])
        return -1;
    else
        return 0;
}

/* Comparator for sorting by sel_est ascending */
static int cmp_sel_est_asc(const void *a, const void *b) {
    const ErrorProfileSample *pa = (const ErrorProfileSample *) a;
    const ErrorProfileSample *pb = (const ErrorProfileSample *) b;
    if (pa->sel_est < pb->sel_est) return -1;
    if (pa->sel_est > pb->sel_est) return 1;
    return 0;
}

/* Comparator for sorting doubles ascending (for sampled vals) */
static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *) a;
    double db = *(const double *) b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Safe log ratio: log(max(sel_true,EPS)/max(sel_est,EPS)) */
static inline double safe_log_ratio(double sel_true, double sel_est) {
    double t = fmax(sel_true, EPS);
    double e = fmax(sel_est, EPS);
    return log(t / e);
}

/* ---------------- RNG (LCG + Box-Muller) ----------------
   Uses the same LCG update as sample_from_distribution.
   - rng_uniform01(): uniform in [0,1)
   - rng_std_normal(): standard normal N(0,1) via Box–Muller
*/
static inline unsigned int lcg_next(unsigned int s) {
    return s * 1664525u + 1013904223u;
}

static inline double rng_uniform01(unsigned int *seed) {
    /* Generate u in [0,1). We mimic your style (mod 1e6) to keep consistency. */
    double u = (double) (*seed % 1000000u) / 1000000.0;
    *seed = lcg_next(*seed);
    return u;
}

static double rng_std_normal(unsigned int *seed) {
    /* Box–Muller transform */
    double u1 = rng_uniform01(seed);
    double u2 = rng_uniform01(seed);
    if (u1 <= 0.0) u1 = 1e-12; /* guard log(0) */
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    return r * cos(theta); /* one normal draw */
}

/* ---------------- KDE helpers (deterministic, no RNG) ---------------- */

/* Evaluate Gaussian KDE density at z:
   f(z) = (1/(n*h*sqrt(2*pi))) * sum_i exp(-0.5*((z - x_i)/h)^2) */
static double kde_eval_gaussian(const double *x, int n, double h, double z) {
    if (n <= 0 || h <= 0.0) return 0.0;
    const double inv_h = 1.0 / h;
    const double norm = 1.0 / (n * h * sqrt(2.0 * M_PI));
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double u = (z - x[i]) * inv_h;
        s += exp(-0.5 * u * u);
    }
    return norm * s;
}

/* ===== Discretize KDE by SAMPLING (no grid; writes into fixed arrays) =====
   Steps:
   1) Draw M samples y_k from KDE:
      - pick i ~ Uniform{0..n-1}
      - y_k = x[i] + h * Z, Z ~ N(0,1)
   2) Sort y_k increasingly.
   3) Estimate local width w_k:
        w0 = y1 - y0; w_{M-1} = y_{M-1} - y_{M-2};
        wk = 0.5 * (y_{k+1} - y_{k-1})
      clip to small positive if needed.
   4) Weight pk ∝ f(y_k) * w_k; normalize to sum=1.
   Output: out->vals=y_k; out->probs=pk; out->sample_count=M_used.
*/
static void discretize_kde_by_sampling(
    const double *x, int n, double h,
    int sample_count,
    unsigned int *seed,
    Distribution *out
) {
    out->sample_count = 0;

    if (n <= 0 || sample_count <= 0) return;
    if (h <= 0.0) h = 1e-6;

    /* cap to fixed capacity */
    int M = sample_count;
    if (M > DIST_MAX_SAMPLE) M = DIST_MAX_SAMPLE;

    /* 1) draw samples */
    double *samples = (double *) malloc(sizeof(double) * M);
    if (!samples) return;

    for (int k = 0; k < M; ++k) {
        double u = rng_uniform01(seed);
        int idx = (int) (u * n);
        if (idx >= n) idx = n - 1;
        double z = rng_std_normal(seed);
        samples[k] = x[idx] + h * z;
    }

    /* 2) sort samples */
    qsort(samples, (size_t)M, sizeof(double), cmp_double_asc);

    /* 3) local widths */
    double *width = (double *) malloc(sizeof(double) * M);
    if (!width) {
        free(samples);
        return;
    }

    if (M == 1) {
        width[0] = 1.0;
    } else {
        width[0] = samples[1] - samples[0];
        width[M - 1] = samples[M - 1] - samples[M - 2];
        for (int k = 1; k < M - 1; ++k) {
            width[k] = 0.5 * (samples[k + 1] - samples[k - 1]);
        }
        for (int k = 0; k < M; ++k) {
            if (!(width[k] > 0.0)) width[k] = 1e-12;
        }
    }

    /* 4) probs ~ density * local width; write to fixed arrays */
    double sumw = 0.0;
    for (int k = 0; k < M; ++k) {
        double fk = kde_eval_gaussian(x, n, h, samples[k]);
        double wk = fk * width[k];
        out->vals[k] = samples[k];
        out->probs[k] = wk;
        sumw += wk;
    }

    /* normalize probabilities */
    if (sumw <= 0.0 || !isfinite(sumw)) {
        for (int k = 0; k < M; ++k) out->probs[k] = 1.0 / (double) M;
    } else {
        for (int k = 0; k < M; ++k) out->probs[k] /= sumw;
    }

    out->sample_count = M;

    /* sort indices by prob descending */
    int *idx = palloc(sizeof(int) * M);
    for (int k = 0; k < M; ++k) idx[k] = k;

    g_dist_for_sort = out; /* set global context */
    qsort(idx, M, sizeof(int), cmp_index_desc);

    /* apply permutation */
    double *tmp_probs = palloc(sizeof(double) * M);
    double *tmp_vals = palloc(sizeof(double) * M);
    for (int k = 0; k < M; ++k) {
        tmp_probs[k] = out->probs[idx[k]];
        tmp_vals[k] = out->vals[idx[k]];
    }
    for (int k = 0; k < M; ++k) {
        out->probs[k] = tmp_probs[k];
        out->vals[k] = tmp_vals[k];
    }

    pfree(idx);
    pfree(tmp_probs);
    pfree(tmp_vals);

    free(width);
    free(samples);
}

/* ===================== Build thresholds + params + dists ===================== */

/* Build per-bin distributions and store thresholds/params into ep. */
void calc_error_dist(ErrorProfile *ep) {
    Assert(ep != NULL);
    Assert(ep->sample_count > 0);
    Assert(error_bin_count >= 1 && error_bin_count <= EP_MAX_BIN);

    /* 1) sort by sel_est */
    qsort((void*)ep->data, (size_t)ep->sample_count,
          sizeof(ErrorProfileSample), cmp_sel_est_asc);

    const int N = ep->sample_count;
    int start = 0;
    double prev_hi = -INFINITY;

    for (int b = 0; b < error_bin_count; ++b) {
        /* 2) equal-count binning on sel_est */
        int end = (int) llround(((long long) (b + 1) * (long long) N) / (double) error_bin_count);
        if (end > N) end = N;
        if (b == error_bin_count - 1) end = N;
        int count = end - start;

        double sel_lo = (b == 0) ? -INFINITY : ep->error_dist_thresh[b - 1];
        double sel_hi = (count > 0)
                            ? ep->data[end - 1].sel_est
                            : ((b == 0) ? INFINITY : prev_hi);

        ep->error_dist_thresh[b] = sel_hi;
        prev_hi = sel_hi;

        /* 3) collect x = log(sel_true/sel_est) for this bin */
        double *x = NULL;
        if (count > 0) {
            x = (double *) malloc(sizeof(double) * count);
            Assert(x != NULL);
            for (int i = 0; i < count; ++i) {
                const ErrorProfileSample *s = &ep->data[start + i];
                x[i] = safe_log_ratio(s->sel_true, s->sel_est);
            }
        }

        /* 4) stats of x */
        double mean_x = 0.0, std_x = 0.0;
        if (count > 0) {
            calc_mean_std(x, count, &mean_x, &std_x);
        }

        /* 5) record params */
        double h = (error_sample_kde_bandwidth > 0.0) ? error_sample_kde_bandwidth : 1e-6;
        ep->params[b].bin_index = b;
        ep->params[b].n_points = count;
        ep->params[b].bandwidth_h = h;
        ep->params[b].mean_logratio = mean_x;
        ep->params[b].std_logratio = std_x;
        ep->params[b].sel_est_lo = sel_lo;
        ep->params[b].sel_est_hi = sel_hi;

        /* 6) build per-bin distribution BY SAMPLING (writes into fixed arrays) */
        ep->error_dist[b].sample_count = 0; /* reset */
        if (error_sample_count > 0 && count > 0) {
            /* decorrelate bins a bit */
            unsigned int bin_seed = error_sample_seed ^ (0x9E3779B9u * (unsigned int) b);
            discretize_kde_by_sampling(
                x, count, h,
                error_sample_count,
                &bin_seed,
                &ep->error_dist[b]
            );
        }

        if (x) free(x);
        start = end;
    }

    /* clear remaining bins (if any) */
    for (int b = error_bin_count; b < EP_MAX_BIN; ++b) {
        ep->error_dist_thresh[b] = NAN;
        ep->params[b].bin_index = -1;
        ep->params[b].n_points = 0;
        ep->params[b].bandwidth_h = 0.0;
        ep->params[b].mean_logratio = 0.0;
        ep->params[b].std_logratio = 0.0;
        ep->params[b].sel_est_lo = NAN;
        ep->params[b].sel_est_hi = NAN;
        ep->error_dist[b].sample_count = 0;
        /* vals/probs arrays remain as-is; sample_count=0 means "empty" */
    }
}

/* ------------------------------- Sampling ------------------------------- */

/* Draw one value from a Distribution according to its probabilities.
   - u is generated from the provided seed via a simple LCG update.
   - Returns the last value as a fallback if numerical rounding leaves a small tail. */
double sample_from_distribution(const Distribution *dist, unsigned int *seed) {
    double u = (double) (*seed % 1000000) / 1000000.0; /* simple uniform in [0,1) */
    *seed = *seed * 1664525u + 1013904223u; /* LCG update */
    double cdf = 0.0;
    for (int i = 0; i < dist->sample_count; i++) {
        cdf += dist->probs[i];
        if (u <= cdf) return dist->vals[i];
    }
    return dist->vals[dist->sample_count - 1];
}

/* Optional helper: find bin by sel_est using (thresh[b-1], thresh[b]].
   Returns -1 if none found. */
int find_bin_by_sel_est(const ErrorProfile *ep, double sel_est) {
    if (error_bin_count <= 0) return -1;
    for (int b = 0; b < error_bin_count; ++b) {
        double lo = (b == 0) ? -INFINITY : ep->error_dist_thresh[b - 1];
        double hi = ep->error_dist_thresh[b];
        if (sel_est > lo && sel_est <= hi) return b;
    }
    /* fallback: the last non-empty bin */
    for (int b = error_bin_count - 1; b >= 0; --b) {
        if (ep->params[b].n_points > 0) return b;
    }
    return -1;
}
