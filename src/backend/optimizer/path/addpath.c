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
 * We keep the k *smallest* scores, so top of heap is the *largest* among kept.
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
 *   - A List* containing all pruned paths (NOT kept)
 *   - If nothing is pruned, returns NIL.
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
    const path_score_strategy add_path_func,
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
     * Phase 1: build PathRank array and compute scores via strategy.
     *
     * If a path has zero samples, assign a very large score.
     * -------------------------------------------------------------------- */
    PathRank *rank_arr = palloc(sizeof(PathRank) * cand_count);

    int idx = 0;
    ListCell *lc;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);
        const Sample *startup_cost_sample = path->startup_cost_sample;
        const Sample *total_cost_sample = path->total_cost_sample;

        Assert(startup_cost_sample != NULL);
        Assert(startup_cost_sample->sample_count >= 0 &&
            startup_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        Assert(total_cost_sample != NULL);
        Assert(total_cost_sample->sample_count >= 0 &&
            total_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        int effective = Min(startup_cost_sample->sample_count, total_cost_sample->sample_count);
        effective = Min(effective, sample_count);
        Assert(effective >= 0);

        double score_val;
        if (effective == 0) {
            /* No samples => automatically worst */
            score_val = DBL_MAX;
        } else {
            score_val = add_path_func(startup_cost_sample, total_cost_sample, min_global, effective);
        }

        rank_arr[idx].path = path;
        rank_arr[idx].score = score_val;
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

    /* hsize should be k unless cand_count < k, but k = Min(limit, cand_count) */
    Assert(hsize == k);

    /* --------------------------------------------------------------------
     * Phase 3: output winners (kept). Order not finalized here.
     * -------------------------------------------------------------------- */
    int *winners = palloc(sizeof(int) * k);
    for (int i = 0; i < k; i++) {
        winners[i] = heap_idx[i];
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
        Assert(writer == losers_cnt);

        pfree(selected);
    }

    /* --------------------------------------------------------------------
     * Phase 4: rebuild lists: kept (write back) and dropped (return).
     * -------------------------------------------------------------------- */
    List *kept_list = NIL;
    for (int i = 0; i < k; i++) {
        const PathRank rank = rank_arr[winners[i]];
        Path *keep = rank.path;
        /* Expose computed score on Path for later stages,
         * and for the final output after the DP process. */
        keep->score = rank.score;
        kept_list = lappend(kept_list, keep);
    }

    List *dropped_list = NIL;
    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const PathRank rank = rank_arr[losers[i]];
            Path *drop = rank.path;
            /* do not overwrite keep->score */
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
 * retain_path_by_strategy
 *
 * Choose up to `retain_path_limit` best Paths from `cand_list` according to
 * the user-provided scoring strategy `retain_path_func`, and append them into
 * the joinrel's pathlist (normal or partial depending on `is_partial`).
 *
 * This function does *not* overwrite existing paths in joinrel->pathlist /
 * partial_pathlist — it only appends the selected ones.
 *
 * The remaining (unselected) Paths from `cand_list` are returned as a new List.
 *
 * Memory / Ownership notes:
 *   - `cand_list` list cells are freed here (Path structs are NOT freed).
 *   - Returned list contains the leftover Path pointers.
 *   - `retain_path_func` computes a per-path score. Lower score = better.
 *
 * Preconditions:
 *   - score_sample_final was already computed for joinrel_first.
 *   - sample_count <= DIST_MAX_SAMPLE.
 *
 * Postconditions:
 *   - Top-k winners appended to joinrel->{pathlist | partial_pathlist}.
 *   - Returned List holds remaining paths.
 */
List *
retain_path_by_strategy(
    const PlannerInfo *root,
    const int lev_index,
    const int rel_index,
    List *cand_list,
    const path_score_strategy retain_path_func,
    const int retain_path_limit,
    int sample_count,
    const bool is_partial
) {
    /* If no candidates, nothing to retain; return immediately */
    if (cand_list == NIL || retain_path_limit == 0) {
        return cand_list;
    }

    /* Basic sanity checks */
    Assert(sample_count >= 1);
    Assert(retain_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);
    Assert(cand_list != NULL);

    /* Fetch RelOptInfo for reading baseline and writing survivors */
    const RelOptInfo *joinrel_first =
            (RelOptInfo *) list_nth(root->join_rel_level_first[lev_index], rel_index);
    RelOptInfo *joinrel =
            (RelOptInfo *) list_nth(root->join_rel_level[lev_index], rel_index);

    /* Count candidates */
    const int cand_count = list_length(cand_list);
    if (cand_count <= 0)
        return NIL;

    /*
     * Get global min per-sample baseline from calc_*_score functions
     * (this is needed for scoring).
     */
    Assert(joinrel_first->score_sample_final != NULL);
    const Sample *score_sample = joinrel_first->score_sample_final;

    Assert(score_sample->sample_count >= 0 &&
        score_sample->sample_count <= DIST_MAX_SAMPLE);

    /* Do not exceed available sample dimensions */
    if (sample_count > score_sample->sample_count)
        sample_count = score_sample->sample_count;

    const double *min_global = score_sample->sample;

    /* ----------------------------------------------------------------------
     * Phase 1: compute score for each candidate path
     * Using provided function retain_path_func (lower = better).
     * Zero-sample paths get DBL_MAX score (worst).
     * ---------------------------------------------------------------------- */
    PathRank *rank_arr = palloc(sizeof(PathRank) * cand_count);

    int idx = 0;
    ListCell *lc;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);
        const Sample *startup_cost_sample = path->startup_cost_sample;
        const Sample *total_cost_sample = path->total_cost_sample;

        Assert(startup_cost_sample != NULL);
        Assert(startup_cost_sample->sample_count >= 0 &&
            startup_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        Assert(total_cost_sample != NULL);
        Assert(total_cost_sample->sample_count >= 0 &&
            total_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        int effective = Min(startup_cost_sample->sample_count, total_cost_sample->sample_count);
        effective = Min(effective, sample_count);
        Assert(effective >= 0);

        double score_val;
        if (effective == 0) {
            /* No samples => automatically worst */
            score_val = DBL_MAX;
        } else {
            score_val = retain_path_func(startup_cost_sample, total_cost_sample, min_global, effective);
        }

        rank_arr[idx].path = path;
        rank_arr[idx].score = score_val;
        idx++;
    }
    Assert(idx == cand_count);

    /* ----------------------------------------------------------------------
     * Phase 2: select top-k (smallest score) via fixed MAX-heap
     * Root of heap = worst among currently kept
     * ---------------------------------------------------------------------- */
    const int k = Min(retain_path_limit, cand_count);

    int *heap_idx = palloc(sizeof(int) * Max(1, k));
    int hsize = 0;

    for (int i = 0; i < cand_count; i++) {
        hsize = rank_idx_maxheap_push_topk(heap_idx, hsize, k, i, rank_arr);
    }
    Assert(hsize == k);

    /* ----------------------------------------------------------------------
     * Phase 3: indices of winners and losers
     * We keep winners, drop losers (return them).
     * ---------------------------------------------------------------------- */
    int *winners = palloc(sizeof(int) * k);
    for (int i = 0; i < k; i++)
        winners[i] = heap_idx[i];

    const int losers_cnt = cand_count - k;
    int *losers = palloc(sizeof(int) * losers_cnt);

    if (losers_cnt > 0) {
        bool *selected = palloc0(sizeof(bool) * cand_count);
        for (int i = 0; i < k; i++) {
            selected[winners[i]] = true;
        }

        int writer = 0;
        for (int i = 0; i < cand_count; i++) {
            if (!selected[i])
                losers[writer++] = i;
        }
        Assert(writer == losers_cnt);
        pfree(selected);
    }

    /* ----------------------------------------------------------------------
     * Phase 4: append winners to joinrel pathlists
     * NOTE: score is **not** stored into Path
     * ---------------------------------------------------------------------- */
    if (is_partial) {
        for (int i = 0; i < k; i++) {
            const PathRank rank = rank_arr[winners[i]];
            Path *keep = rank.path;
            /* do not overwrite keep->score */
            joinrel->partial_pathlist = lappend(joinrel->partial_pathlist, keep);
        }
    } else {
        for (int i = 0; i < k; i++) {
            const PathRank rank = rank_arr[winners[i]];
            Path *keep = rank.path;
            /* do not overwrite keep->score */
            joinrel->pathlist = lappend(joinrel->pathlist, keep);
        }
    }

    /* ----------------------------------------------------------------------
     * Phase 5: return losers as new remaining candidate list
     * ---------------------------------------------------------------------- */
    List *dropped_list = NIL;
    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const PathRank rank = rank_arr[losers[i]];
            Path *drop = rank.path;
            /* do not overwrite keep->score */
            dropped_list = lappend(dropped_list, drop);
        }
    }

    /* old list cells freed; Path structs stay alive */
    list_free(cand_list);

    /* ----------------------------------------------------------------------
     * Cleanup and return losers
     * ---------------------------------------------------------------------- */
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
    for (int j = 0; j < sample_count; j++) {
        min_global[j] = DBL_MAX;
    }

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

/*
 * calc_minimum_envelope
 *
 * Given a List of snapshots of join_rel levels (each element is a List**,
 * indexed by level), compute the elementwise minimum across snapshots of
 * the samples stored in RelOptInfo->score_sample_final->sample.
 *
 * The first snapshot in saved_join_rel_levels is used as the initial
 * envelope and is updated in-place; the function returns that same pointer.
 *
 * Assumptions:
 * - saved_join_rel_levels is non-NIL; each element is a List** with at
 *   least levels_needed+1 entries.
 * - For every level in [2, levels_needed], corresponding per-level lists
 *   have the same order and intended 1:1 correspondence.
 * - rel->score_sample_final is non-NULL and has sample_count elements;
 *   all snapshots agree on sample_count.
 *
 * Complexity:
 *   O(M * L * N * S), where M = number of snapshots,
 *   L = number of levels processed, N = rels per level,
 *   S = sample_count.
 *
 * Note:
 * - Because the result is written back into the first snapshot (min_envelope),
 *   callers that need to preserve it must copy beforehand.
 * - The loop iterates over the first snapshot too; this is harmless since
 *   min(x, x) == x.
 */
List **
calc_minimum_envelope(
    List *saved_join_rel_levels,
    const int sample_count,
    const int levels_needed
) {
    ListCell *lc, *lc1, *lc2;
    /* Use the first snapshot as the initial (and final) envelope; updated in-place. */
    List **min_envelope = linitial(saved_join_rel_levels);

    /* Iterate over every saved snapshot of per-level join_rel lists. */
    foreach(lc, saved_join_rel_levels) {
        List **cur_saved_join_rel_level = lfirst(lc);

        /* By convention, levels 0/1 are base rels; start from level 2. */
        for (int lev = 2; lev <= levels_needed; ++lev) {
            /*
             * Pairwise walk the lists at the same level in the current snapshot
             * and in the running minimum envelope. Assumes identical ordering.
             * If lengths differ, forboth stops at the shorter list.
             */
            forboth(lc1, cur_saved_join_rel_level[lev], lc2, min_envelope[lev]) {
                const RelOptInfo *cur_rel = lfirst(lc1);
                const RelOptInfo *min_rel = lfirst(lc2);

                const Sample *cur_rel_sample = cur_rel->score_sample_final;
                Sample *min_rel_sample = min_rel->score_sample_final;

                /* All samples must agree on the number of points. */
                Assert(cur_rel_sample->sample_count == sample_count);
                Assert(min_rel_sample->sample_count == sample_count);

                const double *cur_sample_raw = cur_rel_sample->sample;
                double *min_sample_raw = min_rel_sample->sample;

                /* Elementwise minimum: write back into the envelope. */
                for (int i = 0; i < sample_count; ++i) {
                    min_sample_raw[i] = Min(cur_sample_raw[i], min_sample_raw[i]);
                }
            }
        }
    }
    /* Return the first snapshot pointer, now containing the minimum envelope. */
    return min_envelope;
}

/*
 * sort_pathlist_by_total_cost
 *
 * Sort a Path list by total_cost in ascending order.
 *
 * Since pathlists are typically small (often < 50 elements),
 * an insertion sort is used instead of list_qsort().
 * Insertion sort has lower overhead for tiny lists and preserves
 * stability (keeps original order on ties).
 *
 * Returns a new List with the same Path* pointers in sorted order.
 */
List *
sort_pathlist_by_total_cost(List *pathlist) {
    if (pathlist == NIL) {
        return NIL;
    }

    const int n = list_length(pathlist);
    if (n <= 1) {
        return pathlist;
    }

    /* Copy to array for O(n^2) insertion sort */
    Path **arr = palloc(sizeof(Path *) * n);

    int i = 0;
    ListCell *lc;
    foreach(lc, pathlist) {
        arr[i++] = (Path *) lfirst(lc);
    }

    /* Insertion sort by total_cost ASC */
    for (int j = 1; j < n; j++) {
        Path *key = arr[j];
        const double key_cost = key->total_cost;
        int k = j - 1;

        while (k >= 0 && arr[k]->total_cost > key_cost) {
            arr[k + 1] = arr[k];
            k--;
        }
        arr[k + 1] = key;
    }

    /* Rebuild list in sorted order */
    List *new_list = NIL;
    for (int j = 0; j < n; j++) {
        new_list = lappend(new_list, arr[j]);
    }

    pfree(arr);
    list_free(pathlist); /* free only list cells, not Path objects */

    return new_list;
}
