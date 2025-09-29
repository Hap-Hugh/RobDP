//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
//

#include "optimizer/dist.h"
#include "optimizer/kde.h"
#include "optimizer/optimizer.h"
#include "utils/smem.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* GUC Parameters */
bool enable_rows_dist;
int error_sample_count;
int error_sample_seed;

double clamp01(double sel) {
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
    Index relid
) {
    Assert(relid > 0);
    RangeTblEntry *rte = root->simple_rte_array[relid];
    char *alias = rte->eref->aliasname;
    elog(LOG, "considering relation: %s", alias);
    return pstrdup(alias);
}

char *get_std_alias(
    const PlannerInfo *root,
    Index relid
) {
    Assert(relid > 0);
    RangeTblEntry *rte = root->simple_rte_array[relid];
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
    double mean = sum / n_samples;
    for (int i = 0; i < n_samples; ++i) {
        double diff = mean - array[i];
        sum_of_squares += diff * diff;
    }
    if (n_samples == 1) {
        *res_std = 0.0;
    } else {
        *res_std = sqrt(sum_of_squares / (n_samples - 1));
    }
}

Distribution *make_single_point_dist(
    double val
) {
    Distribution *dist = palloc0(sizeof(Distribution));
    dist->sample_count = 1;

    /* Build a single point distribution */
    dist->probs[0] = 1.0;
    dist->vals[0] = val;

    return dist;
}

Distribution *scale_distribution(
    const Distribution *src,
    double factor
) {
    if (!src) {
        return NULL;
    }
    Distribution *dst = palloc0(sizeof(Distribution));
    if (!dst) {
        return NULL;
    }
    dst->sample_count = src->sample_count;

    for (int i = 0; i < dst->sample_count; ++i) {
        dst->probs[i] = src->probs[i];
        dst->vals[i] = src->vals[i] * factor;
    }
    return dst;
}

/* Build conditional distribution of sel_true given sel_est:
   - source bin stores log-errors: log(sel_true/sel_est)
   - transform values: val = sel_est * exp(log_error)
   - probabilities are copied (non-uniform). */
Distribution *get_conditional_distribution(const ErrorProfile *ep, double sel_est) {
    static Distribution out; /* static buffer; overwritten on each call */
    /* If you prefer PG allocator, replace the above with:
       Distribution *out = (Distribution*) palloc0(sizeof(Distribution));
    */

    /* reset */
    out.sample_count = 0;
    for (int i = 0; i < DIST_MAX_SAMPLE; ++i) {
        out.vals[i] = 0.0;
        out.probs[i] = 0.0;
    }

    if (!ep) return NULL;

    int b = find_bin_by_sel_est(ep, sel_est);
    if (b < 0) return NULL;

    const Distribution *src = &ep->error_dist[b];
    if (src->sample_count <= 0) return NULL;

    int m = src->sample_count;
    if (m > DIST_MAX_SAMPLE) m = DIST_MAX_SAMPLE;

    /* Transform: y = sel_est * exp(log_error) */
    for (int i = 0; i < m; ++i) {
        double log_err = src->vals[i]; /* this is log(sel_true/sel_est) */
        out.vals[i] = sel_est * exp(log_err);
        out.probs[i] = src->probs[i]; /* keep weights */
    }
    out.sample_count = m;

    /* (optional) tiny renormalization to guard against rounding */
    double s = 0.0;
    for (int i = 0; i < m; ++i) s += out.probs[i];
    if (s > 0.0 && fabs(s - 1.0) > 1e-12) {
        for (int i = 0; i < m; ++i) out.probs[i] /= s;
    }

    return &out; /* or 'return out;' if using palloc0 above */
}

void free_distribution(
    Distribution *dist
) {
    if (!dist) {
        return;
    }
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
        int result = fscanf(fp, "%lf %lf", &sel_true, &sel_est);
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
    double sel_est
) {
    /* 0. Prepare fallback rows estimation result.
     * Note: we don't use `rel->rows`, which has been clamped already. */
    double rows_fallback = sel_est * rel->tuples;

    /* 1. Resolve relation aliases (original alias and a standard fallback). */
    const char *alias = get_alias(root, rel->relid);
    const char *alias_fallback = get_std_alias(root, rel->relid);
    elog(LOG, "[rel %s] considering relation rows distribution.", alias);

    /* 2. Allocate an error profile holder and try to populate it from cache. */
    ErrorProfile *ep;
    bool found = get_error_profile(alias, alias_fallback, &ep);

    /* 2.1 If no profile is available, fall back to a degenerate distribution. */
    if (!found) {
        elog(LOG, "[rel %s] no profile is available, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_single_point_dist(rows_fallback);
        return;
    }

    /* 2.2 Debug-print the error profile. */
    for (int i = 0; i < ep->sample_count; ++i) {
        elog(LOG, "[rel %s] est = %g, true = %g.", alias, ep->data[i].sel_est, ep->data[i].sel_true);
    }

    /* 3. Get conditional distribution p(true_sel | sel_est=e0). */
    Distribution *sel_true_dist = get_conditional_distribution(ep, sel_est);

    /* 3.1 Fallback to single point distribution if we fail to build `sel_true_dist`. */
    if (sel_true_dist == NULL) {
        elog(LOG, "[rel %s] failed to build conditional distribution, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_single_point_dist(rows_fallback);
        return;
    }

    /* 3.2 Debug-print the produced distribution. */
    for (int i = 0; i < sel_true_dist->sample_count; ++i) {
        elog(LOG, "[rel %s] prob = %g, val = %g.", alias, sel_true_dist->probs[i], sel_true_dist->vals[i]);
    }

    /* 4. Scale the `sel_true_dist` -- from selectivity distribution to rows distribution. */
    Distribution *rows_dist = scale_distribution(sel_true_dist, rel->tuples);

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
    double rows_fallback = sel_est * outer_rel->rows * inner_rel->rows;

    /* 1. Resolve relation aliases (original alias and a standard fallback). */
    // FIXME: We assume that the first `rinfo` we found `can_join`.
    // FIXME: We should also check `rinfo->hashjoinoperator`.
    char alias[SM_KEY_LEN];
    char alias_fallback[SM_KEY_LEN];
    memset(alias, 0, sizeof(alias));
    memset(alias_fallback, 0, sizeof(alias_fallback));

    ListCell *lc;
    foreach(lc, restrictlist) {
        RestrictInfo *rinfo = lfirst(lc);
        if (!IsA(rinfo->clause, OpExpr))
            continue;

        OpExpr *opexpr = (OpExpr *) rinfo->clause;
        if (list_length(opexpr->args) != 2)
            continue;

        /* Notes: Both sides must be Vars (skip RelabelType etc.). */
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
    bool found = get_error_profile(alias, alias_fallback, &ep);

    /* 2.1 If no profile is available, fall back to a degenerate distribution. */
    if (!found) {
        elog(LOG, "[rel %s] no profile is available, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_single_point_dist(rows_fallback);
        return;
    }

    /* 3. Get conditional distribution p(true_sel | sel_est=e0). */
    Distribution *sel_true_dist = get_conditional_distribution(ep, sel_est);

    /* 3.1 Fallback to single point distribution if we fail to build `sel_true_dist`. */
    if (sel_true_dist == NULL) {
        elog(LOG, "[rel %s] failed to build conditional distribution, using single point distribution.", alias);
        rel->rows = rows_fallback;
        rel->rows_dist = make_single_point_dist(rows_fallback);
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
