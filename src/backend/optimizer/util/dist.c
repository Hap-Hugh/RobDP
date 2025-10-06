//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
// Modified by Xuan Chen on 2025/10/6.
//

#include "optimizer/dist.h"
#include "optimizer/kde.h"
#include "optimizer/optimizer.h"
#include "utils/smem.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* GUC Parameters */
bool enable_rows_dist;
int error_sample_count;
int error_sample_seed;

double clamp01(const double sel) {
    if (sel < 0.0) {
        return 0.0;
    }
    if (sel > 1.0) {
        return 1.0;
    }
    return sel;
}

char *get_alias(
    const PlannerInfo *root,
    const Index relid
) {
    Assert(relid > 0);
    const RangeTblEntry *rte = root->simple_rte_array[relid];
    char *alias = rte->eref->aliasname;
    elog(LOG, "considering relation: %s", alias);
    return pstrdup(alias);
}

char *get_std_alias(
    const PlannerInfo *root,
    const Index relid
) {
    Assert(relid > 0);
    const RangeTblEntry *rte = root->simple_rte_array[relid];
    char *alias = rte->eref->aliasname;
    char *std_alias = pstrdup(alias);
    char *ch = std_alias;
    while (*ch != '\0') {
        if (*ch >= '0' && *ch <= '9') {
            *ch = '\0'; // *ch is a digit, break now
            break;
        }
        ++ch;
    }
    elog(LOG, "considering relation: %s -> %s", alias, std_alias);
    return std_alias;
}

void calc_mean_std(
    const double *array,
    const int n_samples,
    double *res_mean,
    double *res_std
) {
    if (n_samples <= 0) {
        *res_mean = 0.0;
        *res_std = 0.0;
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        sum += array[i];
    }
    double sum_of_squares = 0.0;
    const double mean = sum / n_samples;
    for (int i = 0; i < n_samples; ++i) {
        const double diff = mean - array[i];
        sum_of_squares += diff * diff;
    }
    if (n_samples == 1) {
        *res_std = 0.0;
    } else {
        *res_std = sqrt(sum_of_squares / (n_samples - 1));
    }
}

Distribution *make_dist_by_single_value(
    const double val
) {
    Distribution *dist = palloc0(sizeof(Distribution));
    dist->sample_count = 1;
    dist->probs = palloc0(sizeof(double));
    dist->vals = palloc0(sizeof(double));

    /* Build a single point distribution */
    dist->probs[0] = 1.0;
    dist->vals[0] = val;

    return dist;
}

Distribution *make_dist_by_scale_factor(
    const Distribution *src,
    const double factor
) {
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
    for (int i = 0; i < dst->sample_count; ++i) {
        dst->probs[i] = src->probs[i];
        dst->vals[i] = src->vals[i] * factor;
    }
    return dst;
}

void free_distribution(
    Distribution *dist
) {
    if (!dist) {
        return;
    }
    pfree(dist->probs);
    pfree(dist->vals);
    pfree(dist);
}

int read_error_profile(
    const char *filename,
    ErrorProfile *ep
) {
    if (!filename || filename[0] == '\0' || ep == NULL) {
        elog(WARNING, "Invalid filename");
        return -1;
    }

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        elog(WARNING, "Could not open file %s", filename);
        return -2;
    }

    memset(ep, 0, sizeof(ErrorProfile));
    double *sel_true_array = palloc0(sizeof(double) * EP_MAX_SAMPLE);
    double *sel_est_array = palloc0(sizeof(double) * EP_MAX_SAMPLE);

    int sample_count = 0;
    while (true) {
        double sel_true, sel_est;
        const int result = fscanf(fp, "%lf %lf", &sel_true, &sel_est);
        if (result == EOF) {
            break;
        }
        if (result == 2) {
            if (sample_count >= EP_MAX_SAMPLE) {
                break;
            }
            sel_true = clamp01(sel_true);
            sel_est = clamp01(sel_est);
            ep->data[sample_count].sel_true = sel_true;
            ep->data[sample_count].sel_est = sel_est;
            sel_true_array[sample_count] = sel_true;
            sel_est_array[sample_count] = sel_est;
            ++sample_count;
        } else {
            int c;
            do {
                c = fgetc(fp);
            } while (c != '\n' && c != EOF);
        }
    }
    fclose(fp);
    ep->sample_count = sample_count;

    if (sample_count > 0) {
        double mean_est, std_est, mean_true, std_true;
        calc_mean_std(sel_est_array, sample_count, &mean_est, &std_est);
        calc_mean_std(sel_true_array, sample_count, &mean_true, &std_true);
        ep->std_est = std_est;
        ep->std_true = std_true;
    } else {
        ep->std_est = 0.0;
        ep->std_true = 0.0;
    }

    pfree(sel_est_array);
    pfree(sel_true_array);
    return 0;
}

bool get_error_profile(
    const char *alias,
    const char *alias_fallback,
    ErrorProfile **ep
) {
    // Try the alias + suffix first
    char ep_key[SM_KEY_LEN];
    sprintf(ep_key, "%s%s", alias, ".txt");
    bool found = SessionMemFind(ep_key, ep);

    // If we find the error profile, return immediately
    if (ep != NULL && found) {
        return true;
    }

    // Otherwise, fallback to alias_fallback + suffix
    if (strcmp(alias, alias_fallback) == 0) {
        // We check whether the alias and alias fallback is the same
        return false;
    }
    char ep_key_fallback[SM_KEY_LEN];
    sprintf(ep_key_fallback, "%s%s", alias_fallback, ".txt");
    found = SessionMemFind(ep_key_fallback, ep);

    if (ep != NULL && found) {
        return true;
    }
    return false;
}

void set_baserel_rows_dist(
    const PlannerInfo *root,
    RelOptInfo *rel,
    const double sel_est
) {
    /* 0. Prepare fallback rows estimation result.
     * Note: we don't use `rel->rows`, which has been clamped already. */
    const double rows_fallback = sel_est * rel->tuples;

    /* 1. Resolve relation aliases (original alias and a standard fallback). */
    const char *alias = get_alias(root, rel->relid);
    const char *alias_fallback = get_std_alias(root, rel->relid);
    elog(LOG, "[rel %s] considering relation rows distribution.", alias);

    /* 2. Allocate an error profile holder and try to populate it from cache. */
    ErrorProfile *ep;
    const bool found = get_error_profile(alias, alias_fallback, &ep);

    /* 2.1 If no profile is available, fall back to a degenerate distribution. */
    if (!found) {
        elog(LOG, "[rel %s] no profile is available, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_dist_by_single_value(rows_fallback);
        return;
    }

    /* 2.2 Debug-print the error profile. */
    for (int i = 0; i < ep->sample_count; ++i) {
        elog(LOG, "[rel %s] est = %g, true = %g.", alias, ep->data[i].sel_est, ep->data[i].sel_true);
    }

    /* 3. Build conditional distribution p(true_sel | sel_est=e0). */
    Distribution *sel_true_dist = build_conditional_distribution(
        ep, sel_est, error_sample_count,
        0.0, error_sample_kde_bandwidth, error_sample_seed
    );

    /* 3.1 Fallback to single point distribution if we fail to build `sel_true_dist`. */
    if (sel_true_dist == NULL) {
        elog(LOG, "[rel %s] failed to build conditional distribution, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_dist_by_single_value(rows_fallback);
        return;
    }

    /* 3.2 Debug-print the produced distribution. */
    for (int i = 0; i < sel_true_dist->sample_count; ++i) {
        elog(LOG, "[rel %s] prob = %g, val = %g.", alias, sel_true_dist->probs[i], sel_true_dist->vals[i]);
    }

    /* 4. Scale the `sel_true_dist` -- from selectivity distribution to rows distribution. */
    Distribution *rows_dist = make_dist_by_scale_factor(sel_true_dist, rel->tuples);

    /* 4.1 Done with selectivity distribution. */
    free_distribution(sel_true_dist);

    /* 5. Calculate the expectation of the rows distribution and update the relation's rows estimation. */
    double rows_dist_mean = 0.0;
    for (int i = 0; i < rows_dist->sample_count; ++i) {
        rows_dist_mean += rows_dist->probs[i] * rows_dist->vals[i];
        elog(LOG, "[rel %s] prob = %g, val = %g, acc mean = %g.",
             alias, rows_dist->probs[i], rows_dist->vals[i], rows_dist_mean);
    }

    /* 5.1 Save the rows and rows distribution. */
    elog(LOG, "[rel %s] originally estimated rows: %g -> adjusted rows: %g.", alias, rel->rows, rows_dist_mean);
    rel->rows = rows_dist_mean;
    rel->rows_dist = rows_dist;
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
static int cmp_by_val_asc(const void *a, const void *b) {
    const int ia = *(const int *) a;
    const int ib = *(const int *) b;
    /* The caller passes parallel arrays; compare by vals[ia] vs vals[ib].
     * We stash pointers via a static for simplicity; see wrapper below. */
    extern const double *g_vals_for_sort;
    const double va = g_vals_for_sort[ia];
    const double vb = g_vals_for_sort[ib];
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* Global pointer used only during qsort compare (reentrant w.r.t single thread). */
const double *g_vals_for_sort = NULL;

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
static Distribution *compress_distribution_equal_mass(const Distribution *src, int target_samples) {
    Assert(target_samples >= 1);
    Assert(src && src->sample_count >= 1);

    /* 1) Filter out non-positive probabilities and compute total mass. */
    int n = src->sample_count;
    double total_mass = 0.0;
    for (int i = 0; i < n; ++i)
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
    for (int i = 0; i < n; ++i) {
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
    g_vals_for_sort = src->vals;
    qsort(idx, m, sizeof(int), cmp_by_val_asc);
    g_vals_for_sort = NULL;

    /* 4) Sweep and form K equal-mass bins. */
    Distribution *dst = palloc(sizeof(Distribution));
    dst->sample_count = target_samples;
    dst->probs = palloc(sizeof(double) * target_samples);
    dst->vals = palloc(sizeof(double) * target_samples);

    const double target_bin_mass = 1.0 / (double) target_samples;
    double bin_mass = 0.0;
    double bin_weighted_val = 0.0;
    int out_k = 0;

    /* Normalize on the fly: prob_norm = prob / total_mass */
    for (int t = 0; t < m; t++) {
        const int i = idx[t];
        const double p = src->probs[i] / total_mass;
        const double v = src->vals[i];

        double remaining = p;
        while (remaining > 0.0 && out_k < target_samples) {
            const double capacity = target_bin_mass - bin_mass;
            double take = remaining;
            if (take > capacity) take = capacity;

            bin_mass += take;
            bin_weighted_val += take * v;
            remaining -= take;

            /* Bin completed: flush one output sample. */
            if (bin_mass >= target_bin_mass - 1e-15) {
                /* Numerical guard: ensure positive mass. */
                const double pmass = bin_mass;
                const double vmean = (pmass > 0.0) ? (bin_weighted_val / pmass) : v;

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
        const double last_val = (out_k > 0) ? dst->vals[out_k - 1] : 0.0;
        while (out_k < target_samples) {
            const double pmass = (out_k == target_samples - 1) ? bin_mass : 0.0;
            const double vmean = (bin_mass > 0.0) ? (bin_weighted_val / (bin_mass)) : last_val;
            dst->probs[out_k] = pmass;
            dst->vals[out_k] = vmean;
            out_k++;
            bin_mass = 0.0;
            bin_weighted_val = 0.0;
        }
    } else if (bin_mass > 0.0) {
        /* All K bins emitted but a tiny tail remains; add it to the last bin. */
        dst->probs[target_samples - 1] += bin_mass;
        const double vtail = (bin_weighted_val > 0.0 && bin_mass > 0.0)
                                 ? (bin_weighted_val / bin_mass)
                                 : dst->vals[target_samples - 1];
        /* Recompute last bin's weighted mean with the tail merged. */
        const double p_old = dst->probs[target_samples - 1] - bin_mass;
        const double v_old = dst->vals[target_samples - 1];
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
        const double scale = 1.0 / sum_check;
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
static Distribution *multiply_distributions_for_join(
    const Distribution *outer_rows_dist,
    const Distribution *inner_rows_dist,
    const Distribution *sel_dist
) {
    const int nO = outer_rows_dist->sample_count;
    const int nI = inner_rows_dist->sample_count;
    const int nS = sel_dist->sample_count;
    const int total = nO * nI * nS;

    Distribution *res = palloc(sizeof(Distribution));
    res->sample_count = total;
    res->probs = palloc(sizeof(double) * total);
    res->vals = palloc(sizeof(double) * total);

    int idx = 0;
    for (int i = 0; i < nO; ++i) {
        const double pO = outer_rows_dist->probs[i];
        const double vO = outer_rows_dist->vals[i];
        if (pO <= 0.0 || !isfinite(pO) || !isfinite(vO))
            continue;

        for (int j = 0; j < nI; j++) {
            double pI = inner_rows_dist->probs[j];
            double vI = inner_rows_dist->vals[j];
            if (pI <= 0.0 || !isfinite(pI) || !isfinite(vI))
                continue;

            for (int k = 0; k < nS; k++) {
                const double pS = sel_dist->probs[k];
                const double vS = sel_dist->vals[k];
                if (pS <= 0.0 || !isfinite(pS) || !isfinite(vS))
                    continue;

                const double p = pO * pI * pS;
                const double v = vO * vI * vS;

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
Distribution *join_rows_distribution(
    const Distribution *outer_rows_dist,
    const Distribution *inner_rows_dist,
    const Distribution *sel_dist,
    const int target_samples
) {
    Distribution *raw = multiply_distributions_for_join(
        outer_rows_dist, inner_rows_dist, sel_dist
    );
    Distribution *compressed = compress_distribution_equal_mass(
        raw, target_samples
    );

    /* Free the big intermediate to control memory bloat. */
    if (raw->probs) {
        pfree(raw->probs);
    }
    if (raw->vals) {
        pfree(raw->vals);
    }
    pfree(raw);
    return compressed;
}

void set_joinrel_rows_dist(
    const PlannerInfo *root,
    RelOptInfo *rel,
    const RelOptInfo *outer_rel,
    const RelOptInfo *inner_rel,
    List *restrictlist,
    const double sel_est
) {
    /* 0. Prepare fallback rows estimation result.
     * Note: we don't use `rel->rows`, which has been clamped already. */
    const double rows_fallback = sel_est * outer_rel->rows * inner_rel->rows;

    /* 1. Resolve relation aliases (original alias and a standard fallback). */
    // FIXME: We assume that the first `rinfo` we found `can_join`.
    // FIXME: We should also check `rinfo->hashjoinoperator`.
    char alias[SM_KEY_LEN];
    char alias_fallback[SM_KEY_LEN];
    memset(alias, 0, sizeof(alias));
    memset(alias_fallback, 0, sizeof(alias_fallback));

    ListCell *lc;
    foreach(lc, restrictlist) {
        const RestrictInfo *rinfo = lfirst(lc);
        if (!IsA(rinfo->clause, OpExpr))
            continue;

        const OpExpr *opexpr = (OpExpr *) rinfo->clause;
        if (list_length(opexpr->args) != 2)
            continue;

        /* Notes: Both sides must be Vars (skip RelabelType etc.). */
        Node *l = linitial(opexpr->args);
        Node *r = lsecond(opexpr->args);
        if (!IsA(l, Var) || !IsA(r, Var))
            continue;

        const Var *leftvar = (Var *) l;
        const Var *rightvar = (Var *) r;

        /* Accept either orientation: (outer,left)-(inner,right) or swapped. */
        const bool l_in_outer = bms_is_member(leftvar->varno, outer_rel->relids);
        const bool r_in_inner = bms_is_member(rightvar->varno, inner_rel->relids);
        const bool r_in_outer = bms_is_member(rightvar->varno, outer_rel->relids);
        const bool l_in_inner = bms_is_member(leftvar->varno, inner_rel->relids);

        if ((l_in_outer && r_in_inner) || (r_in_outer && l_in_inner)) {
            const char *left_rel_alias = get_alias(root, leftvar->varno);
            const char *right_rel_alias = get_alias(root, rightvar->varno);
            const char *left_rel_std_alias = get_std_alias(root, leftvar->varno);
            const char *right_rel_std_alias = get_std_alias(root, rightvar->varno);

            /* Canonicalize order to avoid duplicate “A=B” vs “B=A”. */
            if (strcmp(left_rel_alias, right_rel_alias) < 0)
                snprintf(alias, sizeof(alias), "%s=%s", left_rel_alias, right_rel_alias);
            else
                snprintf(alias, sizeof(alias), "%s=%s", right_rel_alias, left_rel_alias);

            elog(LOG, "Alias: %s; Join Key: %s.%d = %s.%d.",
                 alias,
                 left_rel_alias, leftvar->varattno,
                 right_rel_alias, rightvar->varattno);

            /* Canonicalize order to avoid duplicate “A=B” vs “B=A”. */
            if (strcmp(left_rel_std_alias, right_rel_std_alias) < 0)
                snprintf(alias_fallback, sizeof(alias_fallback), "%s=%s", left_rel_std_alias, right_rel_std_alias);
            else
                snprintf(alias_fallback, sizeof(alias_fallback), "%s=%s", right_rel_std_alias, left_rel_std_alias);

            elog(LOG, "Fallback alias: %s; Join Key: %s.%d = %s.%d.",
                 alias_fallback,
                 left_rel_std_alias, leftvar->varattno,
                 right_rel_std_alias, rightvar->varattno);

            break;
        }
    }
    // TODO: Check whether we have a sane alias with its fallback version.
    elog(LOG, "[rel %s] considering relation rows distribution.", alias);

    /* 2. Allocate an error profile holder and try to populate it from cache. */
    ErrorProfile *ep = 0;
    const bool found = get_error_profile(alias, alias_fallback, &ep);

    /* 2.1 If no profile is available, fall back to a degenerate distribution. */
    if (!found) {
        elog(LOG, "[rel %s] no profile is available, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_dist_by_single_value(rows_fallback);
        return;
    }

    /* 3. Build conditional distribution p(true_sel | sel_est=e0). */
    Distribution *sel_true_dist = build_conditional_distribution(
        ep, sel_est, error_sample_count,
        0.0, error_sample_kde_bandwidth, error_sample_seed
    );

    /* 3.1 Fallback to single point distribution if we fail to build `sel_true_dist`. */
    if (sel_true_dist == NULL) {
        elog(LOG, "[rel %s] failed to build conditional distribution, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_dist_by_single_value(rows_fallback);
        return;
    }

    /* 3.2 Debug-print the produced distribution. */
    for (int i = 0; i < sel_true_dist->sample_count; ++i) {
        elog(LOG, "[rel %s] prob = %g, val = %g.", alias, sel_true_dist->probs[i], sel_true_dist->vals[i]);
    }

    /* 4. Push selectivity uncertainty through the join-size model to get rows distribution.
     * Notes: we assume that both outer relation's and inner relation's rows distribution exist. */
    Distribution *rows_dist = join_rows_distribution(
        outer_rel->rows_dist,
        inner_rel->rows_dist,
        sel_true_dist,
        error_sample_count
    );

    /* 4.1 Done with selectivity distribution. */
    free_distribution(sel_true_dist);

    /* 5. Calculate the expectation of the rows distribution and update the relation's rows estimation. */
    double rows_dist_mean = 0.0;
    for (int i = 0; i < rows_dist->sample_count; ++i) {
        rows_dist_mean += rows_dist->probs[i] * rows_dist->vals[i];
        elog(LOG, "[rel %s] prob = %g, val = %g, acc mean = %g.",
             alias, rows_dist->probs[i], rows_dist->vals[i], rows_dist_mean);
    }

    /* 5.1 Save the rows and rows distribution. */
    elog(LOG, "[rel %s] originally estimated rows: %g -> adjusted rows: %g.", alias, rel->rows, rows_dist_mean);
    rel->rows = rows_dist_mean;
    rel->rows_dist = rows_dist;
}
