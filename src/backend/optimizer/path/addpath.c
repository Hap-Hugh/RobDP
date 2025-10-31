//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/20.
// Modified by Xuan Chen on 2025/10/29.
// Modified by Xuan Chen on 2025/10/31.
//

#include "optimizer/addpath.h"
#include "optimizer/pathstrategy.h"
#include <float.h>

/* Forward Declarations (opaque to this file) */
typedef struct Sample Sample;
typedef struct ErrorProfileRaw ErrorProfileRaw;
typedef struct ErrorSampleParams ErrorSampleParams;
typedef struct ErrorProfile ErrorProfile;

/* ----------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------------*/

/* Per-candidate ranking info (only score is relevant here). */
typedef struct PathRank {
    Path *path;
    double score; /* vs. per-type per-sample minima */
} PathRank;

/* --------- Fixed-size max-heap (size<=k) for PathRank indices by score ASC --
 * We keep the k *smallest* penalties, so top of heap is the *largest* among kept.
 * Comparator: greater(a,b) if a should be closer to root in a MAX-heap.
 * Ties broken by Path* pointer for determinism.
 */

static bool
rank_idx_greater_by_score(const int ia, const int ib, const PathRank *arr) {
    const PathRank *a = &arr[ia];
    const PathRank *b = &arr[ib];
    if (a->score > b->score) return true;
    if (a->score < b->score) return false;
    /* deterministic pointer tiebreaker: greater means closer to root */
    return a->path > b->path;
}

static void
rank_idx_maxheap_sift_up(int *heap, int idx, const PathRank *arr) {
    while (idx > 0) {
        const int parent = (idx - 1) >> 1;
        if (!rank_idx_greater_by_score(heap[idx], heap[parent], arr))
            break;
        const int tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}

static void
rank_idx_maxheap_sift_down(int *heap, const int n, int idx, const PathRank *arr) {
    for (;;) {
        const int l = (idx << 1) + 1;
        const int r = l + 1;
        int largest = idx;

        if (l < n && rank_idx_greater_by_score(heap[l], heap[largest], arr))
            largest = l;
        if (r < n && rank_idx_greater_by_score(heap[r], heap[largest], arr))
            largest = r;
        if (largest == idx)
            break;

        const int tmp = heap[largest];
        heap[largest] = heap[idx];
        heap[idx] = tmp;
        idx = largest;
    }
}

/* Push new index; if heap is full (size==k) and new item is better (smaller score),
 * replace root then sift-down. Returns current heap size (<=k).
 */
static int
rank_idx_maxheap_push_topk(int *heap, const int size, const int k, const int idx, const PathRank *arr) {
    if (size < k) {
        heap[size] = idx;
        rank_idx_maxheap_sift_up(heap, size, arr);
        return size + 1;
    }
    /* if new candidate is better (smaller score) than current worst (root), replace */
    if (rank_idx_greater_by_score(heap[0], idx, arr)) {
        /* root worse than idx */
        heap[0] = idx;
        rank_idx_maxheap_sift_down(heap, k, 0, arr);
    }
    return size; /* unchanged */
}

/*
 * add_path_by_strategy
 *
 * From the current pathlist (normal or partial), retain at most
 * add_path_limit paths by the score metric and discard the rest.
 *
 * Return value:
 *   - A List* containing all pruned paths (NOT kept), sorted by
 *     (score ASC, Path* ASC). If nothing is pruned, returns NIL.
 *
 * Notes:
 *   - The kept list is written back to the RelOptInfo in the same
 *     deterministic order. Paths themselves are not freed.
 *   - Requires score_sample_final to be set beforehand.
 */
List *
add_path_by_strategy(
    const PlannerInfo *root,
    const int lev_index,
    const int rel_index,
    const add_path_strategy add_path_func,
    const int add_path_limit,
    int sample_count,
    const bool is_partial
) {
    /* Basic sanity */
    Assert(sample_count >= 1);
    Assert(add_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    /* Fetch rels for this level/index */
    const RelOptInfo *joinrel_first =
            (RelOptInfo *) list_nth(root->join_rel_level_first[lev_index], rel_index);
    RelOptInfo *joinrel =
            (RelOptInfo *) list_nth(root->join_rel_level[lev_index], rel_index);

    /* Pick target candidate list */
    List *cand_list = is_partial ? joinrel->partial_pathlist : joinrel->pathlist;

    /* Nothing to do */
    const int cand_count = list_length(cand_list);
    if (cand_count <= 0) {
        return NIL;
    }

    /*
     * Fetch finalized per-sample baseline scores (global minima across
     * partial + non-partial), already computed by calc_*_score functions.
     */
    Assert(joinrel_first->score_sample_final != NULL);
    const Sample *score_sample = joinrel_first->score_sample_final;

    Assert(score_sample->sample_count >= 0 &&
        score_sample->sample_count <= DIST_MAX_SAMPLE);

    /* Clamp to available baseline length */
    if (sample_count > score_sample->sample_count) {
        sample_count = score_sample->sample_count;
    }

    const double *min_global = score_sample->sample;

    /* --------------------------------------------------------------------
     * Phase 1: build PathRank array and compute penalties vs min_global.
     *
     * score(p) = max_j {
     *     (v - min_global[j])   if v > min_global[j] * DIST_PENALTY_FACTOR
     *     0.0                   otherwise
     * }
     *
     * If a path has zero samples, assign a very large score.
     * -------------------------------------------------------------------- */
    PathRank *rank_arr = palloc(sizeof(PathRank) * cand_count);

    int idx = 0;
    ListCell *lc;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);
        const Sample *total_cost_sample = path->total_cost_sample;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 &&
            ts->sample_count <= DIST_MAX_SAMPLE);

        const int effective = Min(total_cost_sample->sample_count, sample_count);
        Assert(effective >= 0);

        double max_score;
        if (effective == 0) {
            /* Zero samples => make it lose */
            max_score = DBL_MAX;
        } else {
            max_score = add_path_func(NULL, total_cost_sample, min_global, effective);
        }

        rank_arr[idx].path = path;
        rank_arr[idx].score = max_score;
        idx++;
    }
    Assert(idx == cand_count);

    /* --------------------------------------------------------------------
     * Phase 2: select the global top-k (smallest score) with a fixed MAX-heap.
     * -------------------------------------------------------------------- */
    const int k = Min(add_path_limit, cand_count);

    int *heap_idx = palloc(sizeof(int) * Max(1, k));
    int hsize = 0;

    for (int i = 0; i < cand_count; i++) {
        hsize = rank_idx_maxheap_push_topk(heap_idx, hsize, k, i, rank_arr);
    }

    /* hsize should be k unless cand_count < k */
    Assert(hsize == k);

    /* --------------------------------------------------------------------
     * Phase 3: determinize output order for winners (kept).
     * -------------------------------------------------------------------- */
    int *winners = palloc(sizeof(int) * k);
    for (int i = 0; i < k; i++) {
        winners[i] = heap_idx[i];
    }

    /* insertion sort by (score ASC, Path* ASC) */
    for (int i = 1; i < k; i++) {
        const int winner_i = winners[i];
        const double score_i = rank_arr[winner_i].score;
        const Path *path_i = rank_arr[winner_i].path;
        int j = i - 1;

        while (j >= 0) {
            const int winner_j = winners[j];
            const double score_j = rank_arr[winner_j].score;
            const Path *path_j = rank_arr[winner_j].path;

            if (score_i < score_j || (score_i == score_j && path_i < path_j)) {
                winners[j + 1] = winners[j];
                j--;
            } else {
                break;
            }
        }
        winners[j + 1] = winner_i;
    }

    /* Also compute losers (pruned) indices */
    const int losers_cnt = cand_count - k;
    int *losers = palloc(sizeof(int) * losers_cnt);
    if (losers_cnt > 0) {
        bool *selected = palloc0(sizeof(bool) * cand_count);
        for (int i = 0; i < k; i++) {
            selected[winners[i]] = true;
        }

        int writer = 0;
        for (int i = 0; i < cand_count; i++) {
            if (!selected[i]) {
                losers[writer++] = i;
            }
        }
        Assert(m == losers_cnt);

        pfree(selected);
    }

    /* --------------------------------------------------------------------
     * Phase 4: rebuild lists: kept (write back) and dropped (return).
     * -------------------------------------------------------------------- */
    List *kept_list = NIL;
    for (int i = 0; i < k; i++) {
        const PathRank rank = rank_arr[winners[i]];
        Path *keep = rank.path;
        /* optional: expose computed score on Path for later stages */
        keep->score = rank.score;
        kept_list = lappend(kept_list, keep);
    }

    List *dropped_list = NIL;
    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const PathRank rank = rank_arr[losers[i]];
            Path *drop = rank.path;
            drop->score = 0.0;
            dropped_list = lappend(dropped_list, drop);
        }
    }

    /* Free the old list cells; Paths are kept alive. */
    list_free(cand_list);

    /* Write back survivors */
    if (is_partial) {
        joinrel->partial_pathlist = kept_list;
    } else {
        joinrel->pathlist = kept_list;
    }

    /* --------------------------------------------------------------------
     * Phase 5: free temporaries and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_arr);
    pfree(heap_idx);
    pfree(winners);
    pfree(losers);

    return dropped_list;
}

/*
 * calc_score_from_pathlist
 *
 * For each sample index j (0 <= j < sample_count), compute the minimum
 * total_cost_sample[j] across all candidate paths in either:
 *
 *   - joinrel->pathlist            (if is_partial == false)
 *   - joinrel->partial_pathlist    (if is_partial == true)
 *
 * Then store that per-sample minimum vector into:
 *
 *   - joinrel->score_sample        (if is_partial == false)
 *   - joinrel->score_sample_partial(if is_partial == true)
 *
 * Assumptions:
 *   - Every Path in the chosen candidate list has total_cost_sample != NULL.
 *   - total_cost_sample->sample_count <= DIST_MAX_SAMPLE.
 *   - sample_count is bounded by DIST_MAX_SAMPLE as well.
 */
void
calc_score_from_pathlist(
    RelOptInfo *joinrel,
    const int sample_count,
    const bool is_partial
) {
    /* Sanity check */
    Assert(sample_count >= 0 && sample_count <= DIST_MAX_SAMPLE);

    /*
     * Phase 0:
     * Compute per-sample global minima across ALL candidate paths.
     * Initialize an array of length sample_count to DBL_MAX,
     * then take the minimum observed value for each sample index.
     */
    /* Choose which path list to inspect: partial or full */
    List *cand_list = is_partial ? joinrel->partial_pathlist : joinrel->pathlist;

    if (cand_list == NULL) {
        if (is_partial) {
            joinrel->score_sample_partial = NULL;
        } else {
            joinrel->score_sample = NULL;
        }
        return;
    }

    double *min_global = palloc(sizeof(double) * sample_count);
    for (int j = 0; j < sample_count; j++)
        min_global[j] = DBL_MAX;

    ListCell *lc;
    foreach(lc, cand_list) {
        const Path *p = (Path *) lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 && ts->sample_count <= DIST_MAX_SAMPLE);

        /* Only compare up to the smaller of ts->sample_count and sample_count */
        const int effective = Min(ts->sample_count, sample_count);

        for (int j = 0; j < effective; j++) {
            const double v = ts->sample[j];

            if (v < min_global[j]) {
                min_global[j] = v;
            }
        }
    }

    /*
     * Phase 1:
     * Materialize the result into the appropriate score_sample field
     * on the joinrel.
     *
     * We allocate (or reuse if already allocated) the Sample struct,
     * set its length, and copy the values we just computed.
     */
    Sample *score_sample;
    if (is_partial) {
        joinrel->score_sample_partial = (Sample *) palloc(sizeof(Sample));
        score_sample = joinrel->score_sample_partial;
    } else {
        joinrel->score_sample = (Sample *) palloc(sizeof(Sample));
        score_sample = joinrel->score_sample;
    }

    score_sample->sample_count = sample_count;

    for (int j = 0; j < sample_count; j++) {
        score_sample->sample[j] = min_global[j];
    }

    /*
     * Cleanup:
     * min_global was just a scratch buffer in the current memory context.
     * pfree() is not strictly required in planner code if the context
     * is short-lived, but it's good hygiene.
     */
    pfree(min_global);
}

/*
 * calc_final_score_from_pathlist
 *
 * Combine per-sample cost scores from:
 *   - joinrel->score_sample          (normal paths)
 *   - joinrel->score_sample_partial  (partial paths), if any
 *
 * If no partial score exists, simply use score_sample as the final result.
 * Otherwise, for each sample index j, take the minimum of:
 *   score_sample->sample[j]
 *   score_sample_partial->sample[j]
 *
 * Preconditions:
 *   - score_sample != NULL
 *   - If score_sample_partial exists, both sample arrays must have
 *     the same positive sample_count.
 *
 * Output:
 *   - joinrel->score_sample_final
 */
void
calc_final_score_from_pathlist(
    RelOptInfo *joinrel
) {
    const Sample *score_sample = joinrel->score_sample;
    const Sample *score_sample_partial = joinrel->score_sample_partial;

    /* Must have at least normal paths' score */
    if (score_sample == NULL) {
        elog(ERROR, "score_sample is NULL in calc_final_score_from_pathlist");
    }

    /* No partial path scores: final = deep copy of normal */
    if (score_sample_partial == NULL) {
        const int count = score_sample->sample_count;
        Sample *final = palloc(sizeof(Sample));
        final->sample_count = count;

        for (int j = 0; j < count; j++) {
            final->sample[j] = score_sample->sample[j];
        }

        joinrel->score_sample_final = final;
        return;
    }

    /* Sanity checks for partial scores */
    if (score_sample->sample_count <= 0 ||
        score_sample_partial->sample_count <= 0 ||
        score_sample->sample_count != score_sample_partial->sample_count) {
        elog(ERROR, "inconsistent sample_count: normal=%d partial=%d",
             score_sample->sample_count, score_sample_partial->sample_count);
    }

    /* Combine by per-sample minimum */
    const int count = score_sample->sample_count;
    Sample *final = palloc(sizeof(Sample));
    final->sample_count = count;

    for (int j = 0; j < count; j++) {
        const double v1 = score_sample->sample[j];
        const double v2 = score_sample_partial->sample[j];
        final->sample[j] = (v1 < v2) ? v1 : v2;
    }

    joinrel->score_sample_final = final;
}
