//
// Created by Xuan Chen on 2025/9/15.
//

#include "postgres.h"
#include "c.h"
#include "optimizer/dist.h"
#include "optimizer/optimizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
static double maxd(double a, double b) {
    return a > b ? a : b;
}

static double mind(double a, double b) {
    return a < b ? a : b;
}

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
 * Result compression strategy
 * ---------------------------
 * We compress an arbitrary discrete distribution down to K samples by
 * "equal-probability binning":
 *   1) Normalize probabilities to sum to 1.0 (skip non-positive probs).
 *   2) Sort samples by 'val' ascending.
 *   3) Cut the CDF into K equal-mass bins (target mass ~= 1/K per bin).
 *   4) For each bin, output one representative sample whose:
 *        - prob  = sum of probs in the bin
 *        - val   = probability-weighted mean value within the bin
 *   5) Renormalize tiny numerical drift so probs sum exactly to 1.0.
 *
 * This preserves the CDF shape better than naive top-K or uniform-value binning,
 * and avoids exploding sample_count after multiplications.
 */
static int
cmp_by_val_asc(const void *a, const void *b) {
    const int ia = *(const int *) a;
    const int ib = *(const int *) b;
    /* The caller passes parallel arrays; compare by vals[ia] vs vals[ib].
     * We stash pointers via a static for simplicity; see wrapper below. */
    extern const double *__g_vals_for_sort;
    double va = __g_vals_for_sort[ia];
    double vb = __g_vals_for_sort[ib];
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* Global pointer used only during qsort compare (reentrant w.r.t single thread). */
const double *__g_vals_for_sort = NULL;

/*
 * compress_distribution_equal_mass
 * --------------------------------
 * Reduce 'src' to exactly 'target_samples' points using equal-probability
 * (quantile) binning. Returns a freshly palloc'd Distribution.
 *
 * Requirements:
 *   - target_samples >= 1
 *   - src may contain arbitrary sample_count >= 1
 */
static Distribution *
compress_distribution_equal_mass(const Distribution *src, int target_samples) {
    Assert(target_samples >= 1);
    Assert(src && src->sample_count >= 1);

    /* 1) Filter out non-positive probabilities and compute total mass. */
    int n = src->sample_count;
    double total_mass = 0.0;
    for (int i = 0; i < n; i++)
        if (src->probs[i] > 0.0 && isfinite(src->probs[i]) && isfinite(src->vals[i]))
            total_mass += src->probs[i];

    /* Edge case: if no positive mass, fall back to a single zero sample. */
    if (total_mass <= 0.0) {
        Distribution *empty = palloc(sizeof(Distribution));
        empty->sample_count = 1;
        empty->probs = palloc(sizeof(double));
        empty->vals = palloc(sizeof(double));
        empty->probs[0] = 1.0;
        empty->vals[0] = 0.0;
        return empty;
    }

    /* 2) Build index array of valid samples and normalize probs. */
    int *idx = palloc(sizeof(int) * n);
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (src->probs[i] > 0.0 && isfinite(src->probs[i]) && isfinite(src->vals[i]))
            idx[m++] = i;
    }
    if (m == 0) {
        /* Should not happen due to total_mass check, but keep safe. */
        Distribution *empty = palloc(sizeof(Distribution));
        empty->sample_count = 1;
        empty->probs = palloc(sizeof(double));
        empty->vals = palloc(sizeof(double));
        empty->probs[0] = 1.0;
        empty->vals[0] = 0.0;
        pfree(idx);
        return empty;
    }

    /* 3) Sort by value ascending via index indirection. */
    __g_vals_for_sort = src->vals;
    qsort(idx, m, sizeof(int), cmp_by_val_asc);
    __g_vals_for_sort = NULL;

    /* 4) Sweep and form K equal-mass bins. */
    Distribution *dst = palloc(sizeof(Distribution));
    dst->sample_count = target_samples;
    dst->probs = palloc(sizeof(double) * target_samples);
    dst->vals = palloc(sizeof(double) * target_samples);

    double target_bin_mass = 1.0 / (double) target_samples;
    double carry_mass = 0.0;
    double bin_mass = 0.0;
    double bin_weighted_val = 0.0;
    int out_k = 0;

    /* Normalize on the fly: prob_norm = prob / total_mass */
    for (int t = 0; t < m; t++) {
        int i = idx[t];
        double p = src->probs[i] / total_mass;
        double v = src->vals[i];

        double remaining = p;
        while (remaining > 0.0 && out_k < target_samples) {
            double capacity = target_bin_mass - bin_mass;
            double take = remaining;
            if (take > capacity) take = capacity;

            bin_mass += take;
            bin_weighted_val += take * v;
            remaining -= take;

            /* Bin completed: flush one output sample. */
            if (bin_mass >= target_bin_mass - 1e-15) {
                /* Numerical guard: ensure positive mass. */
                double pmass = bin_mass;
                double vmean = (pmass > 0.0) ? (bin_weighted_val / pmass) : v;

                dst->probs[out_k] = pmass;
                dst->vals[out_k] = vmean;
                out_k++;

                /* Reset accumulators for next bin. */
                bin_mass = 0.0;
                bin_weighted_val = 0.0;
            }
        }
    }

    /* If we have leftover mass (due to rounding), assign it to the last bin. */
    if (out_k < target_samples) {
        /* Fill any missing bins by cloning the last known value with zero mass. */
        double last_val = (out_k > 0) ? dst->vals[out_k - 1] : 0.0;
        while (out_k < target_samples) {
            double pmass = (out_k == target_samples - 1) ? bin_mass : 0.0;
            double vmean = (bin_mass > 0.0) ? (bin_weighted_val / (bin_mass)) : last_val;
            dst->probs[out_k] = pmass;
            dst->vals[out_k] = vmean;
            out_k++;
            bin_mass = 0.0;
            bin_weighted_val = 0.0;
        }
    } else if (bin_mass > 0.0) {
        /* All K bins emitted but a tiny tail remains; add it to the last bin. */
        dst->probs[target_samples - 1] += bin_mass;
        double vtail = (bin_weighted_val > 0.0 && bin_mass > 0.0)
                           ? (bin_weighted_val / bin_mass)
                           : dst->vals[target_samples - 1];
        /* Recompute last bin's weighted mean with the tail merged. */
        double p_old = dst->probs[target_samples - 1] - bin_mass;
        double v_old = dst->vals[target_samples - 1];
        if (dst->probs[target_samples - 1] > 0.0) {
            dst->vals[target_samples - 1] =
                    (p_old * v_old + bin_mass * vtail) / dst->probs[target_samples - 1];
        }
    }

    /* 5) Final renormalization to fix FP drift: force sum(probs) == 1.0 */
    double sum_check = 0.0;
    for (int k = 0; k < target_samples; k++)
        sum_check += dst->probs[k];
    if (sum_check > 0.0 && fabs(sum_check - 1.0) > 1e-12) {
        double scale = 1.0 / sum_check;
        for (int k = 0; k < target_samples; k++)
            dst->probs[k] *= scale;
    }

    pfree(idx);
    return dst;
}

/*
 * multiply_distributions_for_join
 * --------------------------------
 * Triple product convolution:
 *   rows = outer_rows * inner_rows * selectivity
 * Produces a (potentially large) intermediate Distribution.
 */
static Distribution *
multiply_distributions_for_join(const Distribution *outer_rows_dist,
                                const Distribution *inner_rows_dist,
                                const Distribution *sel_dist) {
    int nO = outer_rows_dist->sample_count;
    int nI = inner_rows_dist->sample_count;
    int nS = sel_dist->sample_count;
    int total = nO * nI * nS;

    Distribution *res = palloc(sizeof(Distribution));
    res->sample_count = total;
    res->probs = palloc(sizeof(double) * total);
    res->vals = palloc(sizeof(double) * total);

    int idx = 0;
    for (int i = 0; i < nO; i++) {
        double pO = outer_rows_dist->probs[i];
        double vO = outer_rows_dist->vals[i];
        if (pO <= 0.0 || !isfinite(pO) || !isfinite(vO))
            continue;

        for (int j = 0; j < nI; j++) {
            double pI = inner_rows_dist->probs[j];
            double vI = inner_rows_dist->vals[j];
            if (pI <= 0.0 || !isfinite(pI) || !isfinite(vI))
                continue;

            for (int k = 0; k < nS; k++) {
                double pS = sel_dist->probs[k];
                double vS = sel_dist->vals[k];
                if (pS <= 0.0 || !isfinite(pS) || !isfinite(vS))
                    continue;

                double p = pO * pI * pS;
                double v = vO * vI * vS;

                res->probs[idx] = p;
                res->vals[idx] = v;
                idx++;
            }
        }
    }

    /* If some samples were skipped due to invalids, shrink arrays. */
    if (idx < total) {
        res->sample_count = idx;
        res->probs = repalloc(res->probs, sizeof(double) * idx);
        res->vals = repalloc(res->vals, sizeof(double) * idx);
    }

    return res;
}

/*
 * join_rows_distribution
 * ----------------------
 * Convenience wrapper:
 *   1) Multiply three input distributions (outer, inner, selectivity).
 *   2) Compress the resulting distribution to 'target_samples' points.
 */
Distribution *
join_rows_distribution(const Distribution *outer_rows_dist,
                       const Distribution *inner_rows_dist,
                       const Distribution *sel_dist,
                       int target_samples) {
    Distribution *raw = multiply_distributions_for_join(outer_rows_dist,
                                                        inner_rows_dist,
                                                        sel_dist);
    Distribution *compressed = compress_distribution_equal_mass(raw, target_samples);

    /* Free the big intermediate to control memory bloat. */
    if (raw) {
        if (raw->probs) pfree(raw->probs);
        if (raw->vals) pfree(raw->vals);
        pfree(raw);
    }
    return compressed;
}

/*
 * Load an error profile from "<base_dir>/<alias>.txt".
 * Each line is: <true_selectivity> <estimated_selectivity>
 * Returns 0 on success. On failure, returns a nonzero code and leaves *out zeroed.
 *
 * Populates:
 *  - out->data: array of EPSample { true_sel, est_sel } (palloc0'ed)
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
    EPSample *buf = (EPSample *) palloc0(cap * sizeof(EPSample));
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
                pfree(buf);
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
        pfree(buf);
        return 5;
    }

    out->data = buf;
    out->n = n;

    /* Compute stddevs for estimated and true selectivities. */
    double *ests = (double *) palloc0(n * sizeof(double));
    double *trus = (double *) palloc0(n * sizeof(double));
    if (!ests || !trus) {
        pfree(ests);
        pfree(trus);
        pfree(buf);
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

    pfree(ests);
    pfree(trus);
    return 0;
}

void set_baserel_rows_dist(
    PlannerInfo *root,
    RelOptInfo *rel,
    char *error_profile_path,
    double est_sel
) {
    ErrorProfile *ep = palloc0(sizeof(ErrorProfile));
    RangeTblEntry *rte = root->simple_rte_array[rel->relid];
    elog(LOG, "considering relation %s", rte->eref->aliasname);

    /* Load the error profile. */
    if (load_error_profile(error_profile_path, rte->eref->aliasname, ep) != 0) {
        elog(LOG, "failed to load error profile for relation %s, using single point distribution.",
             rte->eref->aliasname);

        /* Here we encode the current point-estimate selectivity as a degenerate dist. */
        rel->rows_dist = make_single_point_dist(est_sel * rel->tuples);
        return; /* Nothing more to do for the dist path. */
    }

    /* We would like to save the error profile for future use. */
    rel->sel_error_profile = ep;

    /* Build a conditional distribution p(true_sel | est_sel=e0). */
    const double e0 = est_sel; /* current point estimate as the condition */
    const int n_samples = 20; /* TODO: consider a GUC */
    double h_est = 0.0; /* out-params if supported by builder */
    double h_true = 0.0;

    /*
     * Now we have estimated selectivity and error profile, we can calculate
     * the distribution of the true selectivity (`true_sel_dist`).
     */
    Distribution *true_sel_dist = build_conditional_distribution(
        ep, e0, n_samples, h_est, h_true, 42
    );

    if (!true_sel_dist) {
        elog(LOG, "failed to build conditional distribution for relation %s, using single point distribution.",
             rte->eref->aliasname);
        rel->rows_dist = make_single_point_dist(est_sel * rel->tuples);
        return;
    }

    /* Debug-print produced samples (value = sel, prob = weight). */
    for (int i = 0; i < true_sel_dist->sample_count; i++) {
        elog(DEBUG1, "rows_dist(sel) sample: %g (p = %g)",
             true_sel_dist->vals[i], true_sel_dist->probs[i]);
    }

    rel->rows_dist = scale_distribution(true_sel_dist, rel->tuples);
    double dist_mean = 0.0;
    for (int i = 0; i < rel->rows_dist->sample_count; i++) {
        dist_mean += rel->rows_dist->probs[i] * rel->rows_dist->vals[i];
        elog(DEBUG1, "%g %g %g", rel->rows_dist->probs[i], rel->rows_dist->vals[i], dist_mean);
    }
    elog(LOG, "rows original: %g -> rows_dist mean: %g",
         rel->rows, dist_mean);
    rel->rows = clamp_row_est(dist_mean);
    free_distribution(true_sel_dist);
}

void set_joinrel_rows_dist(
    PlannerInfo *root,
    RelOptInfo *rel,
    RelOptInfo *outer_rel,
    RelOptInfo *inner_rel,
    SpecialJoinInfo *sjinfo,
    List *restrictlist,
    char *error_profile_path
) {
    /* --- Try to build a filename key for the join error profile --- */
    char filename[64];
    MemSet(filename, 0, sizeof(filename));

    ListCell *lc;
    foreach(lc, restrictlist) {
        RestrictInfo *rinfo = (RestrictInfo *) lfirst(lc);
        if (!IsA(rinfo->clause, OpExpr))
            continue;

        OpExpr *opexpr = (OpExpr *) rinfo->clause;
        if (list_length(opexpr->args) != 2)
            continue;

        /* Be defensive: both sides must be Vars (skip RelabelType etc.). */
        Node *l = linitial(opexpr->args);
        Node *r = lsecond(opexpr->args);
        if (!IsA(l, Var) || !IsA(r, Var))
            continue;

        Var *leftvar = (Var *) l;
        Var *rightvar = (Var *) r;

        /* Accept either orientation: (outer,left)-(inner,right) or swapped. */
        bool l_in_outer = bms_is_member(leftvar->varno, outer_rel->relids);
        bool r_in_inner = bms_is_member(rightvar->varno, inner_rel->relids);
        bool r_in_outer = bms_is_member(rightvar->varno, outer_rel->relids);
        bool l_in_inner = bms_is_member(leftvar->varno, inner_rel->relids);

        if ((l_in_outer && r_in_inner) || (r_in_outer && l_in_inner)) {
            const char *left = root->simple_rte_array[leftvar->varno]->eref->aliasname;
            const char *right = root->simple_rte_array[rightvar->varno]->eref->aliasname;

            /* Canonicalize order to avoid duplicate “A=B” vs “B=A”. */
            if (strcmp(left, right) < 0)
                snprintf(filename, sizeof(filename), "%s=%s", left, right);
            else
                snprintf(filename, sizeof(filename), "%s=%s", right, left);

            elog(LOG, "Filename: %s; Join Key: %s.%d = %s.%d",
                 filename,
                 root->simple_rte_array[leftvar->varno]->eref->aliasname, leftvar->varattno,
                 root->simple_rte_array[rightvar->varno]->eref->aliasname, rightvar->varattno);
            break;
        }
    }

    /* --- Estimated selectivity for conditioning p(true_sel | est_sel=e0) --- */
    double est_sel = clauselist_selectivity(
        root,
        restrictlist,
        0, /* varRelid=0 for joins */
        sjinfo->jointype, /* FIX: respect actual jointype (was JOIN_INNER) */
        sjinfo
    );

    /* --- Load error profile and build conditional selectivity distribution --- */
    ErrorProfile *ep = palloc0(sizeof(ErrorProfile));

    if (filename[0] == '\0' ||
        load_error_profile(error_profile_path, filename, ep) != 0) {
        /* No suitable key or load failed: fall back to a degenerate ROWS dist. */
        if (filename[0] == '\0')
            elog(LOG, "no equi-join key found; using single-point rows distribution.");
        else
            elog(LOG, "failed to load error profile for %s; using single-point rows distribution.", filename);

        rel->rows_dist = make_single_point_dist(rel->rows);
        pfree(ep); /* avoid leaking the shell struct */
        return;
    }

    /* Keep the profile for later (must be freed with free_error_profile + pfree). */
    rel->sel_error_profile = ep;

    /* Build a conditional distribution p(true_sel | est_sel=e0). */
    const double e0 = est_sel; /* current point estimate as the condition */
    const int n_samples = 20; /* TODO: consider a GUC */
    double h_est = 0.0; /* out-params if supported by builder */
    double h_true = 0.0;

    /*
     * Now we have estimated selectivity and error profile, we can calculate
     * the distribution of the true selectivity (`true_sel_dist`).
     */
    Distribution *true_sel_dist = build_conditional_distribution(
        ep, e0, n_samples, h_est, h_true, 42
    );

    if (!true_sel_dist) {
        elog(LOG, "failed to build conditional selectivity distribution; using single-point rows distribution.");
        rel->rows_dist = make_single_point_dist(rel->rows);
        return;
    }

    /* Debug: these are SELECTIVITY samples. */
    for (int i = 0; i < true_sel_dist->sample_count; i++)
        elog(DEBUG1, "rows_dist(sel) sample: %g (p = %g)",
         true_sel_dist->vals[i], true_sel_dist->probs[i]);

    /* Push selectivity uncertainty through the join-size model to get ROWS dist. */
    rel->rows_dist = join_rows_distribution(
        outer_rel->rows_dist, inner_rel->rows_dist, true_sel_dist, n_samples
    );

    /* Done with selectivity samples. */
    free_distribution(true_sel_dist);

    /* Update rel->rows to the mean of the rows distribution. */
    double dist_mean = 0.0;
    for (int i = 0; i < rel->rows_dist->sample_count; i++) {
        dist_mean += rel->rows_dist->probs[i] * rel->rows_dist->vals[i];
        elog(DEBUG1, "%g %g %g", rel->rows_dist->probs[i], rel->rows_dist->vals[i], dist_mean);
    }

    elog(LOG, "rows original: %g -> rows_dist mean: %g", rel->rows, dist_mean);
    rel->rows = clamp_row_est(dist_mean);
}

/* Free storage inside an ErrorProfile (safe on partially-filled structs). */
void free_error_profile(ErrorProfile *ep) {
    if (!ep) return;
    pfree(ep->data);
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
    Pair *arr = (Pair *) palloc0(n * sizeof(Pair));
    if (!arr) return 0;

    for (int i = 0; i < n; i++) {
        arr[i].d = fabs(ep->data[i].est_sel - e0);
        arr[i].i = i;
    }
    qsort(arr, n, sizeof(Pair), cmp);

    int m = (k < n) ? k : n;
    for (int j = 0; j < m; j++) out_idx[j] = arr[j].i;

    pfree(arr);
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
        double *tmp = (double *) palloc0(ep->n * sizeof(double));
        if (!tmp) return NULL;
        for (int i = 0; i < ep->n; i++) tmp[i] = ep->data[i].est_sel;

        double mean, std;
        mean_std(tmp, ep->n, &mean, &std);
        double iqr = estimate_iqr(tmp, ep->n);
        pfree(tmp);

        double robust_sigma = (iqr > 0) ? mind(std, iqr / 1.349) : std;
        h_est = silverman_bandwidth(robust_sigma, ep->n);
        h_est = maxd(h_est, 1e-3);
    }

    /* ---------- 2) Precompute log-errors: eps_i = log(T_i) - log(E_i) ---------- */
    const double EPS = 1e-15; /* avoid log(0). Tune if needed. */
    int n = ep->n;
    double *errs = (double *) palloc0(n * sizeof(double));
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
        double *tmp2 = (double *) palloc0(n * sizeof(double));
        if (!tmp2) {
            pfree(errs);
            return NULL;
        }
        for (int i = 0; i < n; i++) tmp2[i] = errs[i];

        double iqr = estimate_iqr(tmp2, n);
        pfree(tmp2);

        double robust_sigma = (iqr > 0) ? mind(std, iqr / 1.349) : std;
        /* Smaller jitter than KDE bandwidth; feel free to tune factor (e.g., 0.5) */
        h_true = 0.25 * silverman_bandwidth(robust_sigma, n);
        h_true = maxd(h_true, 1e-3);
    }

    /* ---------- 4) Kernel weights in est_sel axis (unchanged) ---------- */
    double *w = (double *) palloc0(n * sizeof(double));
    if (!w) {
        pfree(errs);
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
        int *idx = (int *) palloc0(k * sizeof(int));
        if (!idx) {
            pfree(w);
            pfree(errs);
            return NULL;
        }
        int m = knn_indices_by_est(ep, e0, k, idx);

        for (int i = 0; i < n; i++) w[i] = 0.0;
        for (int j = 0; j < m; j++) w[idx[j]] = 1.0;
        sumw = (double) m;

        pfree(idx);
    }

    /* ---------- 5) Build CDF for inverse-CDF sampling (unchanged) ---------- */
    double *cdf = (double *) palloc0(n * sizeof(double));
    if (!cdf) {
        pfree(w);
        pfree(errs);
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
    Distribution *dist = (Distribution *) palloc0(sizeof(Distribution));
    if (!dist) {
        pfree(cdf);
        pfree(w);
        pfree(errs);
        return NULL;
    }

    dist->sample_count = n_samples;
    dist->probs = (double *) palloc0(n_samples * sizeof(double));
    dist->vals = (double *) palloc0(n_samples * sizeof(double));
    if (!dist->probs || !dist->vals) {
        pfree(dist->probs);
        pfree(dist->vals);
        pfree(dist);
        pfree(cdf);
        pfree(w);
        pfree(errs);
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

    pfree(cdf);
    pfree(w);
    pfree(errs);
    return dist;
}

Distribution *scale_distribution(const Distribution *src, double factor) {
    if (!src) {
        return NULL;
    }
    Distribution *dst = palloc0(sizeof(Distribution));
    if (!dst) {
        return NULL;
    }
    dst->sample_count = src->sample_count;
    dst->probs = palloc0(dst->sample_count * sizeof(double));
    dst->vals = palloc0(dst->sample_count * sizeof(double));

    if (!dst->probs || !dst->vals) {
        pfree(dst->probs);
        pfree(dst->vals);
        pfree(dst);
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
    pfree(dist->probs);
    pfree(dist->vals);
    pfree(dist);
}

Distribution *make_single_point_dist(double val) {
    Distribution *dist = palloc0(sizeof(Distribution));
    dist->sample_count = 1;
    dist->probs = palloc0(sizeof(double));
    dist->vals = palloc0(sizeof(double));

    /* Single Point distribution */
    dist->probs[0] = 1.0;
    dist->vals[0] = val;

    return dist;
}
