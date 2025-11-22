//
// Created by Xuan Chen on 2025/11/21.
//

/*
 * calc_plan_similarity.c
 *
 * Jensen–Shannon based plan-space reduction via greedy K-center,
 * following the Python implementation of:
 *
 *   k_center_greedy(self.costCollection, R, JS_distance, first_plan=None)
 *
 * For each Path in cand_list we assume there is a per-sample total
 * cost vector stored in a Sample struct:
 *
 *   struct Sample {
 *       int    sample_count;
 *       double sample[DIST_MAX_SAMPLE];   // log-based total cost per sample
 *   };
 *
 * and that each Path has a pointer to such a Sample. Adjust the field
 * access (path->samples) to your actual custom Path type.
 */

#include "postgres.h"

#include <math.h>
#include <float.h>

#include "nodes/pg_list.h"
#include "lib/stringinfo.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "optimizer/pathstrategy.h"

/*
 * Number of centers (R) to select. In a real implementation this would
 * typically be defined as a GUC; here we declare it as an extern so
 * you can define it elsewhere.
 */
static int plan_similarity_k = 5;

/* ----------------------------------------------------------------
 * Internal helpers: random int, KL, JS, etc.
 * ----------------------------------------------------------------
 */

/*
 * Return a random integer in [lo, hi], inclusive.
 * Replace this with whatever RNG you prefer in PostgreSQL
 * (e.g., pg_prng_uint64_range) if you need strict determinism.
 */
static int
random_int_between(int lo, int hi) {
    long r;

    Assert(lo <= hi);

    if (lo == hi)
        return lo;

    /* simple wrapper; you can improve this if needed */
    r = random(); /* uses libc random(), seeded elsewhere */
    return lo + (int) (r % (hi - lo + 1));
}

/*
 * Compute KL(p || q) where p and q are raw non-negative vectors
 * of length sample_count. This matches the Python logic:
 *
 *   - Normalize p and q separately to sum to 1.
 *   - Add a small epsilon to avoid log(0).
 *
 * We do not require p and q to be pre-normalized; we treat them
 * as arbitrary non-negative cost vectors.
 */
static double
kl_divergence(const double *p, const double *q, int sample_count) {
    const double EPS = 1e-10;
    double sum_p = 0.0;
    double sum_q = 0.0;
    double kl = 0.0;
    int i;

    /* 1) Compute sums */
    for (i = 0; i < sample_count; i++) {
        sum_p += p[i];
        sum_q += q[i];
    }

    /* Guard against degenerate all-zero vectors */
    if (sum_p <= 0.0 || sum_q <= 0.0)
        return 0.0;

    /* 2) Normalize, add epsilon, and accumulate KL */
    for (i = 0; i < sample_count; i++) {
        double pi = p[i] / sum_p;
        double qi = q[i] / sum_q;

        pi += EPS;
        qi += EPS;

        kl += pi * log(pi / qi);
    }

    return kl;
}

/*
 * Jensen–Shannon distance between two cost vectors p and q.
 *
 * This mirrors the Python implementation:
 *
 *   m = 0.5 * (p + q)
 *   js_div = 0.5 * KL(p || m) + 0.5 * KL(q || m)
 *   return sqrt(js_div)
 *
 * Note: p and q are raw non-negative cost vectors; each KL()
 * call normalizes its inputs internally, just like the Python code.
 */
static double
js_distance(const double *p, const double *q, int sample_count) {
    double *m;
    double js_div;
    double result;
    int i;

    m = (double *) palloc(sizeof(double) * sample_count);

    /* Compute mixture distribution m = 0.5 * (p + q) */
    for (i = 0; i < sample_count; i++)
        m[i] = 0.5 * (p[i] + q[i]);

    js_div = 0.5 * kl_divergence(p, m, sample_count)
             + 0.5 * kl_divergence(q, m, sample_count);

    if (js_div < 0.0)
        js_div = 0.0;

    result = sqrt(js_div);

    pfree(m);

    return result;
}

/* ----------------------------------------------------------------
 * Main function: JS-based K-center over Path cost samples
 * ----------------------------------------------------------------
 */

/*
 * calc_plan_similarity
 *
 * Use Jensen–Shannon distance over per-sample total cost vectors
 * to perform greedy K-center selection on the candidate Paths,
 * mirroring the Python code:
 *
 *   centers, assignments = k_center_greedy(costCollection, R, JS_distance, first_plan=None)
 *
 * The result is encoded in rank_arr[].score:
 *
 *   - score = 1.0 for center plans (selected representatives)
 *   - score = 0.0 for non-center plans (to be pruned if caller wishes)
 *
 * Caller is expected to use rank_arr to filter/sort candidates.
 *
 * min_envelope is currently unused here; it is kept in the signature
 * for compatibility and for potential future regret-based metrics.
 */
extern void
calc_plan_similarity(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope, /* unused for JS-based similarity */
    const int sample_count
) {
    int cand_count;
    int i, k;
    ListCell *lc;
    Path **path_array; /* indexable view of cand_list */
    double **cost_matrix; /* [cand_count][sample_count] */
    int *centers; /* indices of selected center plans */
    double *min_dist; /* per-plan distance to nearest center */
    int num_centers; /* R: number of centers we want */

    (void) min_envelope; /* silence unused-variable warning for now */

    /* ----------------------------------------------------------
     * 0. Basic setup and sanity checks
     * ----------------------------------------------------------
     */

    cand_count = list_length((List *) cand_list);
    if (cand_count <= 0) {
        elog(LOG, "calc_plan_similarity: empty candidate list");
        return;
    }

    /* Determine number of centers R; clamp to [1, cand_count]. */
    num_centers = plan_similarity_k;
    if (num_centers <= 0)
        num_centers = 1;
    if (num_centers > cand_count)
        num_centers = cand_count;

    elog(LOG,
         "calc_plan_similarity: cand_count=%d, sample_count=%d, target_centers=%d",
         cand_count, sample_count, num_centers);

    /* ----------------------------------------------------------
     * 1. Materialize cand_list into an indexable array
     *    and fill rank_arr.path
     * ----------------------------------------------------------
     */

    path_array = (Path **) palloc(sizeof(Path *) * cand_count);

    i = 0;
    foreach(lc, cand_list) {
        Path *path = (Path *) lfirst(lc);

        path_array[i] = path;
        rank_arr[i].path = path;
        rank_arr[i].score = 0.0; /* will be filled after K-center */
        i++;
    }
    Assert(i == cand_count);

    /* ----------------------------------------------------------
     * 2. Build cost_matrix: one cost vector per Path
     *    (this corresponds to self.costCollection in Python)
     * ----------------------------------------------------------
     *
     * cost_matrix[i][s] = total cost of plan i at sample s.
     * We assume each Path has a Sample* with at least sample_count
     * entries. Adjust Path->samples access as needed.
     */

    cost_matrix = (double **) palloc(sizeof(double *) * cand_count);

    for (i = 0; i < cand_count; i++) {
        Sample *sm = path_array[i]->total_cost_sample;
        int sc;

        if (sm == NULL)
            elog(ERROR, "calc_plan_similarity: path %d has no Sample data", i);

        cost_matrix[i] = (double *) palloc(sizeof(double) * sample_count);

        /*
         * Case 1: exact match → copy directly
         */
        if (sm->sample_count == sample_count) {
            for (sc = 0; sc < sample_count; sc++)
                cost_matrix[i][sc] = sm->sample[sc];
        }
        /*
         * Case 2: only 1 sample → broadcast that single value
         */
        else if (sm->sample_count == 1) {
            double v = sm->sample[0];
            for (sc = 0; sc < sample_count; sc++)
                cost_matrix[i][sc] = v;

            elog(LOG,
                 "calc_plan_similarity: path %d only has 1 sample; broadcasting to %d slots",
                 i, sample_count);
        }
        /*
         * Case 3: illegal → insufficient samples but not 1
         */
        else {
            elog(ERROR,
                 "calc_plan_similarity: path %d has %d samples, required %d "
                 "(only sample_count==1 is allowed for broadcasting)",
                 i, sm->sample_count, sample_count);
        }
    }

    /* ----------------------------------------------------------
     * 3. Run greedy K-center on cost_matrix using JS distance
     *    (this mirrors k_center_greedy(costCollection, R, JS_distance))
     * ----------------------------------------------------------
     */

    centers = (int *) palloc(sizeof(int) * num_centers);
    min_dist = (double *) palloc(sizeof(double) * cand_count);

    /* 3.1 Initialize: pick the first center randomly
     * (Python behavior when first_plan=None).
     */
    {
        int first_center;

        first_center = random_int_between(0, cand_count - 1);
        centers[0] = first_center;

        for (i = 0; i < cand_count; i++)
            min_dist[i] = DBL_MAX;

        elog(LOG, "calc_plan_similarity: first center is candidate %d", first_center);
    }

    /* 3.2 Iteratively select the remaining centers */
    for (k = 1; k < num_centers; k++) {
        int last_center = centers[k - 1];
        int farthest_idx = -1;
        double farthest_dist = -1.0;

        /* Update each candidate's distance to the nearest selected center */
        for (i = 0; i < cand_count; i++) {
            double d;

            d = js_distance(cost_matrix[i], cost_matrix[last_center], sample_count);

            if (d < min_dist[i])
                min_dist[i] = d;

            /* Track the point farthest from its closest center */
            if (min_dist[i] > farthest_dist) {
                farthest_dist = min_dist[i];
                farthest_idx = i;
            }
        }

        centers[k] = farthest_idx;

        elog(LOG,
             "calc_plan_similarity: selected center #%d -> candidate %d (min_dist=%.6f)",
             k, farthest_idx, farthest_dist);
    }

    /* ----------------------------------------------------------
     * 4. Assign scores based on whether the path is selected
     *    as a center.
     *
     * Python code *drops* non-center plans by slicing arrays:
     *   self.costCollection   = [...]
     *   self.penaltyCollection = [...]
     *   self.plan_list        = [...]
     *
     * Here we encode the same semantics via scores:
     *   - center plans:   score = 1.0
     *   - non-center:     score = 0.0
     *
     * The caller can then keep only center plans (score > 0).
     * ----------------------------------------------------------
     */

    /* Default: all non-centers */
    for (i = 0; i < cand_count; i++)
        rank_arr[i].score = 0.0;

    /* Mark selected centers */
    for (k = 0; k < num_centers; k++) {
        int center_idx = centers[k];

        Assert(center_idx >= 0 && center_idx < cand_count);
        rank_arr[center_idx].score = 1.0;
    }

    elog(LOG,
         "calc_plan_similarity: reduced from %d candidates to %d centers",
         cand_count, num_centers);

    /* Optional: log the list of chosen centers */
    {
        StringInfoData buf;

        initStringInfo(&buf);
        appendStringInfoString(&buf, "calc_plan_similarity: centers = [");
        for (k = 0; k < num_centers; k++) {
            if (k > 0)
                appendStringInfoString(&buf, ", ");
            appendStringInfo(&buf, "%d", centers[k]);
        }
        appendStringInfoString(&buf, "]");
        elog(LOG, "%s", buf.data);

        pfree(buf.data);
    }

    /* ----------------------------------------------------------
     * 5. Cleanup temporary structures (optional in short-lived
     *    contexts, but good practice in long-lived sessions)
     * ----------------------------------------------------------
     */

    for (i = 0; i < cand_count; i++)
        pfree(cost_matrix[i]);

    pfree(cost_matrix);
    pfree(centers);
    pfree(min_dist);
    pfree(path_array);
}
