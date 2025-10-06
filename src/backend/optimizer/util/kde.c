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

/* GUC Parameters */
double error_sample_kde_bandwidth = 0.0;

/* ---------- Types ---------- */

/* (distance, index) pair for sorting by distance while remembering original i */
typedef struct {
    double dist;
    int idx;
} DistIndexPair;

/* ---------- Small helpers ---------- */

static double maxd(double a, double b) {
    return (a > b) ? a : b;
}

static double mind(double a, double b) {
    return (a < b) ? a : b;
}

/*
 * Silverman's rule-of-thumb for 1D KDE bandwidth:
 *   h = 1.06 * sigma * n^(-1/5)
 * Falls back to a small positive value if n < 2 or sigma <= 0.
 */
static double silverman_bandwidth(double sigma, int n) {
    if (n < 2 || sigma <= 0.0)
        return 1e-3;
    return 1.06 * sigma * pow(n, -0.2);
}

/* qsort comparator for doubles (ascending). */
static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/*
 * Estimate IQR (Q3 - Q1) using simple index-based quartiles after sorting.
 * Mutates the given array by sorting it in-place.
 */
static double estimate_iqr(double *tmp, int n) {
    qsort(tmp, n, sizeof(double), cmp_double);

    /* Use simple linear indices; robust enough for bandwidth purposes. */
    const int q1i = (int) floor(0.25 * (n - 1));
    const int q3i = (int) floor(0.75 * (n - 1));
    return tmp[q3i] - tmp[q1i];
}

/*
 * Gaussian kernel (unnormalized): K(u) = exp(-0.5*u^2).
 * We don’t need the normalization constant for relative weights.
 */
static double gaussian_kernel_u(double u) {
    return exp(-0.5 * u * u);
}

/* Uniform(0, 1) via rand_r(), avoiding exact 0/1 endpoints. */
static double uniform01(unsigned int *state) {
    /* +1 / +2 trick keeps result in (0, 1), not including endpoints. */
    return (rand_r(state) + 1.0) / ((double) RAND_MAX + 2.0);
}

/*
 * Box–Muller normal(0, 1) using rand_r()-based uniform(0, 1).
 * Reentrant w.r.t. the passed-in PRNG state.
 */
static double randn(unsigned int *state) {
    const double u1 = uniform01(state);
    const double u2 = uniform01(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* qsort comparator for DistIndexPair by ascending distance. */
static int cmp_pair_by_d(const void *a, const void *b) {
    const double x = ((const DistIndexPair *) a)->dist;
    const double y = ((const DistIndexPair *) b)->dist;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/*
 * Return indices of the k nearest neighbors in 'ep' w.r.t. |est_sel - e0|.
 * Writes into out_idx[0..k-1]. Returns the number of indices written (<= k).
 * Complexity: O(n log n) due to sorting; fine for moderate n.
 */
static int knn_indices_by_est(const ErrorProfile *ep, double e0, int k, int *out_idx) {
    const int n = ep->sample_count;
    DistIndexPair *arr = palloc0(n * sizeof(DistIndexPair));

    for (int i = 0; i < n; ++i) {
        arr[i].dist = fabs(ep->data[i].sel_est - e0);
        arr[i].idx = i;
    }
    qsort(arr, n, sizeof(DistIndexPair), cmp_pair_by_d);

    const int m = (k < n) ? k : n;
    for (int j = 0; j < m; ++j)
        out_idx[j] = arr[j].idx;

    pfree(arr);
    return m;
}

/* =========================================================================
 * Public entry: build_conditional_distribution()
 * -------------------------------------------------------------------------
 * Build an empirical Distribution of true selectivity conditioned on an
 * estimate est_sel, using:
 *   - KDE weights on the estimate axis
 *   - log-error jittering in log true/est space
 *
 * If h_est <= 0, it is chosen via robust Silverman rule on {sel_est}.
 * If h_true <= 0, it is chosen via a shrunk Silverman rule on {log-errors}.
 *
 * Returns a Distribution* with n_samples values, each with equal mass.
 * On (rare) degenerate cases (e.g., est_sel far outside support), falls back
 * to a KNN mask before building the CDF.
 * =========================================================================
 */
Distribution *build_conditional_distribution(
    const ErrorProfile *ep,
    const double est_sel,
    const int n_samples,
    double h_est,
    double h_true,
    unsigned int seed
) {
    if (ep == NULL || ep->sample_count <= 0 || n_samples <= 0)
        return NULL;

    const double EPS = 1e-15;
    const int n = ep->sample_count;

    /* ---------- 1) Bandwidth on estimate axis (robust Silverman) ---------- */
    if (h_est <= 0.0) {
        double *tmp = palloc0(n * sizeof(double));
        for (int i = 0; i < n; ++i)
            tmp[i] = ep->data[i].sel_est;

        double mean = 0.0, std = 0.0;
        calc_mean_std(tmp, n, &mean, &std);

        /* estimate_iqr sorts in-place, so reuse tmp directly */
        const double iqr = estimate_iqr(tmp, n);
        const double robust_sigma = (iqr > 0.0) ? mind(std, iqr / 1.349) : std;

        h_est = silverman_bandwidth(robust_sigma, n);
        h_est = maxd(h_est, 1e-3);

        pfree(tmp);
    }

    /* ---------- 2) Precompute per-sample log-errors: log(T) - log(E) ------ */
    double *errs = palloc0(n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        const double T = ep->data[i].sel_true;
        const double E = ep->data[i].sel_est;

        /* Clamp only to avoid non-positive arguments to log(). */
        const double Tc = (T <= 0.0) ? EPS : T;
        const double Ec = (E <= 0.0) ? EPS : E;

        errs[i] = log(Tc) - log(Ec);
    }

    /* ---------- 3) Bandwidth on log-error axis (if not provided) ---------- */
    if (h_true <= 0.0) {
        double mean = 0.0, std = 0.0;
        calc_mean_std(errs, n, &mean, &std);

        /* Build a scratch copy for IQR (estimate_iqr sorts in-place). */
        double *tmp2 = palloc0(n * sizeof(double));
        for (int i = 0; i < n; ++i)
            tmp2[i] = errs[i];

        const double iqr = estimate_iqr(tmp2, n);
        const double robust_sigma = (iqr > 0.0) ? mind(std, iqr / 1.349) : std;

        /* Slightly smaller than Silverman’s: tune factor as desired (e.g., 0.25). */
        h_true = 0.25 * silverman_bandwidth(robust_sigma, n);
        h_true = maxd(h_true, 1e-3);

        pfree(tmp2);
    }

    /* ---------- 4) Kernel weights on estimate axis ------------------------ */
    double *w = palloc0(n * sizeof(double));

    double sumw = 0.0;
    const double inv_h = 1.0 / h_est;

    for (int i = 0; i < n; ++i) {
        const double u = (ep->data[i].sel_est - est_sel) * inv_h;
        const double wi = gaussian_kernel_u(u);
        w[i] = wi;
        sumw += wi;
    }

    /*
     * KNN fallback if weights are vanishingly small (e.g., est_sel far outside
     * support or h_est too tiny). Use ~max(50, sqrt(n)) neighbors.
     */
    if (sumw < 1e-12) {
        const int k = (int) fmin(n, fmax(50.0, sqrt(n)));
        int *idx = palloc0(k * sizeof(int));
        const int m = knn_indices_by_est(ep, est_sel, k, idx);

        /* Convert to a simple mask over indices returned by KNN. */
        for (int i = 0; i < n; ++i)
            w[i] = 0.0;
        for (int j = 0; j < m; ++j)
            w[idx[j]] = 1.0;
        sumw = (double) m;

        pfree(idx);
    }

    /* ---------- 5) Build CDF over sample weights -------------------------- */
    double *cdf = palloc0(n * sizeof(double));

    if (sumw > 0.0) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            acc += w[i] / sumw;
            cdf[i] = acc;
        }
        cdf[n - 1] = 1.0; /* ensure exact 1.0 at the end */
    } else {
        /* Shouldn’t happen due to the KNN fallback, but keep a guard. */
        for (int i = 0; i < n; ++i)
            cdf[i] = (i + 1) / (double) n;
    }

    /* ---------- 6) Allocate output Distribution --------------------------- */
    Distribution *dist = palloc0(sizeof(Distribution));
    dist->sample_count = n_samples;
    dist->probs = (double *) palloc0(n_samples * sizeof(double));
    dist->vals = (double *) palloc0(n_samples * sizeof(double));

    /* ---------- 7) Draw samples via inverse-CDF + log-error jitter -------- */
    for (int j = 0; j < n_samples; ++j) {
        /* Inverse-CDF pick. */
        const double u = uniform01(&seed);

        int lo = 0, hi = n - 1;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (cdf[mid] < u)
                lo = mid + 1;
            else
                hi = mid;
        }
        const int pick = lo;

        /* Add Normal(0, h_true^2) jitter in log-error space. */
        const double err_draw = errs[pick] + h_true * randn(&seed);
        const double t_draw = est_sel * exp(err_draw);

        /* Clamp to [0, 1]; for est_sel=0, this naturally collapses to 0. */
        dist->vals[j] = clamp01(t_draw);
        dist->probs[j] = 1.0 / (double) n_samples;
    }

    /* ---------- 8) Cleanup and return ------------------------------------- */
    pfree(cdf);
    pfree(w);
    pfree(errs);

    return dist;
}
