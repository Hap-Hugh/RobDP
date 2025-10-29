//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/20.
// Modified by Xuan Chen on 2025/10/29.
//

#include "optimizer/addpath.h"
#include <float.h>

/* Forward Declarations (opaque to this file) */
typedef struct Sample Sample;
typedef struct ErrorProfileRaw ErrorProfileRaw;
typedef struct ErrorSampleParams ErrorSampleParams;
typedef struct ErrorProfile ErrorProfile;

/* ----------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------------*/

/* Per-candidate ranking info (only penalty is relevant here). */
typedef struct PathRank {
    Path *path;
    double max_penalty; /* vs. per-type per-sample minima */
} PathRank;

/* ---------------- Min-heap for Path* by total_cost ASC (pointer tiebreak) --- */

static inline bool
path_less_total_cost(const Path *a, const Path *b) {
    if (a->total_cost < b->total_cost) return true;
    if (a->total_cost > b->total_cost) return false;
    /* deterministic pointer tiebreaker */
    return a < b;
}

static void
path_minheap_sift_down(Path **heap, int n, int idx) {
    for (;;) {
        int l = (idx << 1) + 1;
        int r = l + 1;
        int smallest = idx;

        if (l < n && path_less_total_cost(heap[l], heap[smallest]))
            smallest = l;
        if (r < n && path_less_total_cost(heap[r], heap[smallest]))
            smallest = r;
        if (smallest == idx)
            break;

        Path *tmp = heap[smallest];
        heap[smallest] = heap[idx];
        heap[idx] = tmp;
        idx = smallest;
    }
}

static Path *
path_minheap_pop(Path **heap, int *pn) {
    int n = *pn;
    Assert(n > 0);
    Path *ret = heap[0];
    heap[0] = heap[n - 1];
    *pn = n - 1;
    if (*pn > 0)
        path_minheap_sift_down(heap, *pn, 0);
    return ret;
}

/* --------- Fixed-size max-heap (size<=k) for PathRank indices by penalty ASC --
 * We keep the k *smallest* penalties, so top of heap is the *largest* among kept.
 * Comparator: greater(a,b) if a should be closer to root in a MAX-heap.
 * Ties broken by Path* pointer for determinism.
 */

static inline bool
rank_idx_greater_by_penalty(int ia, int ib, const PathRank *arr) {
    const PathRank *a = &arr[ia];
    const PathRank *b = &arr[ib];
    if (a->max_penalty > b->max_penalty) return true;
    if (a->max_penalty < b->max_penalty) return false;
    /* deterministic pointer tiebreaker: greater means closer to root */
    return a->path > b->path;
}

static void
rankidx_maxheap_sift_up(int *heap, int idx, const PathRank *arr) {
    while (idx > 0) {
        int parent = (idx - 1) >> 1;
        if (!rank_idx_greater_by_penalty(heap[idx], heap[parent], arr))
            break;
        int tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}

static void
rankidx_maxheap_sift_down(int *heap, int n, int idx, const PathRank *arr) {
    for (;;) {
        int l = (idx << 1) + 1;
        int r = l + 1;
        int largest = idx;

        if (l < n && rank_idx_greater_by_penalty(heap[l], heap[largest], arr))
            largest = l;
        if (r < n && rank_idx_greater_by_penalty(heap[r], heap[largest], arr))
            largest = r;
        if (largest == idx)
            break;

        int tmp = heap[largest];
        heap[largest] = heap[idx];
        heap[idx] = tmp;
        idx = largest;
    }
}

/* Push new index; if heap is full (size==k) and new item is better (smaller penalty),
 * replace root then sift-down. Returns current heap size (<=k).
 */
static int
rankidx_maxheap_push_topk(int *heap, int size, int k, int idx, const PathRank *arr) {
    if (size < k) {
        heap[size] = idx;
        rankidx_maxheap_sift_up(heap, size, arr);
        return size + 1;
    }
    /* if new candidate is better (smaller penalty) than current worst (root), replace */
    if (rank_idx_greater_by_penalty(heap[0], idx, arr)) {
        /* root worse than idx */
        heap[0] = idx;
        rankidx_maxheap_sift_down(heap, k, 0, arr);
    }
    return size; /* unchanged */
}

/* Build a min-heap in O(n) from an unsorted array. */
static void
path_minheap_heapify(Path **heap, int n) {
    if (n <= 1)
        return;

    /* last internal node is (n-2)/2 */
    for (int i = (n - 2) >> 1; i >= 0; i--)
        path_minheap_sift_down(heap, n, i);
}

/*
 * reconsider_pathlist
 *
 * Replace *pathlist_ptr (either normal or partial) with at most
 * mp_path_limit paths chosen from the current list.
 *
 * We score/rank each Path by its "penalty":
 *
 *   penalty(p) = max_j ( sample_p[j] - min_global[j] )
 *
 * where:
 *   - sample_p[j]        is p->total_cost_sample->sample[j]
 *   - min_global[j]      is the global minimum cost at sample j
 *                         that was previously computed and stored in
 *                         joinrel->score_sample (or score_sample_partial)
 *
 * Only the first sample_count samples are considered.
 *
 * Then:
 *   - We keep the mp_path_limit paths with the SMALLEST penalties.
 *   - For determinism, the final list is sorted by
 *          (penalty ASC, Path* ASC)
 *
 * Finally:
 *   - We rebuild the RelOptInfo's {pathlist | partial_pathlist} in
 *     that deterministic order, and discard the others.
 *
 * Notes:
 *   - We do NOT free the Path nodes themselves, only the List cell
 *     structure that held the losing ones.
 *   - We assume rankidx_maxheap_push_topk() exists and implements a
 *     fixed-size max-heap for "best k so far".
 */
void
reconsider_pathlist(
    PlannerInfo *root,
    int lev_index,
    int rel_index,
    int sample_count,
    int mp_path_limit,
    bool is_partial
) {
    RelOptInfo *joinrel;
    RelOptInfo *joinrel_first;
    List *candlist;
    Sample *score_sample;
    double *min_global;
    ListCell *lc;

    PathRank *rank_arr;
    int cand_count;
    int idx;
    int k;
    int *heap_idx;
    int hsize;
    int *winners;

    /* Basic sanity */
    Assert(sample_count >= 1);
    Assert(mp_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    /*
     * Grab the joinrel at this (lev_index, rel_index).
     * Note: list_nth() returns a void*, cast to RelOptInfo*.
     */
    joinrel_first = (RelOptInfo *) list_nth(root->join_rel_level_first[lev_index], rel_index);
    joinrel = (RelOptInfo *) list_nth(root->join_rel_level[lev_index], rel_index);

    /* Pick which candidate list to operate on based on is_partial */
    candlist = is_partial
                   ? joinrel->partial_pathlist
                   : joinrel->pathlist;

    /* If there is nothing to consider, just bail out. */
    cand_count = list_length(candlist);
    if (cand_count <= 0)
        return;

    /*
     * Fetch the global minima ("baseline") for scoring.
     *
     * NOTE: The global minima for partial paths and non-partial
     *       paths may differ, because they could have been computed
     *       separately (score_sample_partial vs score_sample).
     *
     * We assume that calc_score_from_pathlist() was already called
     * to populate these.
     */
    if (is_partial) {
        Assert(joinrel_first->score_sample_partial != NULL);
        score_sample = joinrel_first->score_sample_partial;
    } else {
        Assert(joinrel_first->score_sample != NULL);
        score_sample = joinrel_first->score_sample;
    }

    Assert(score_sample->sample_count >= 0 &&
        score_sample->sample_count <= DIST_MAX_SAMPLE);

    /*
     * We only compare up to sample_count samples, but also should not read
     * more than score_sample->sample_count.
     */
    if (sample_count > score_sample->sample_count)
        sample_count = score_sample->sample_count;

    min_global = score_sample->sample;

    /* --------------------------------------------------------------------
     * Phase 1: build PathRank array and compute penalties vs min_global.
     *
     * path penalty = max over j of (path_sample[j] - min_global[j])
     * If a path has zero samples, we assign it a very large penalty
     * so that it will likely lose.
     * -------------------------------------------------------------------- */
    rank_arr = (PathRank *) palloc(sizeof(PathRank) * cand_count);

    idx = 0;
    foreach(lc, candlist) {
        Path *p = (Path *) lfirst(lc);
        const Sample *ts = p->total_cost_sample;
        int effective;
        double max_pen;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 &&
            ts->sample_count <= DIST_MAX_SAMPLE);

        /* Only compare up to the min of both sample counts */
        effective = Min(ts->sample_count, sample_count);

        if (effective <= 0) {
            /* No samples? Penalize heavily. */
            max_pen = DBL_MAX;
        } else {
            max_pen = -DBL_MAX;

            for (int j = 0; j < effective; j++) {
                const double pen = ts->sample[j] - min_global[j];
                if (pen > max_pen)
                    max_pen = pen;
            }
        }

        rank_arr[idx].path = p;
        rank_arr[idx].max_penalty = max_pen;
        idx++;
    }
    Assert(idx == cand_count);

    /* --------------------------------------------------------------------
     * Phase 2: select the global top-k (smallest penalty).
     *
     * We keep only mp_path_limit paths. We use a fixed-size MAX-heap
     * of indices into rank_arr. The heap root is always the *worst*
     * (highest penalty) among the current kept set, so we can quickly
     * evict it if we find a strictly better candidate.
     * -------------------------------------------------------------------- */
    k = Min(mp_path_limit, cand_count);

    heap_idx = (int *) palloc(sizeof(int) * Max(1, k));
    hsize = 0;

    for (int i = 0; i < cand_count; i++) {
        /*
         * rankidx_maxheap_push_topk(heap_idx, hsize, k, i, rank_arr)
         *
         * Contract (expected):
         *   - Inserts candidate index i into the "best k so far" structure.
         *   - If size < k, just push.
         *   - Else if rank_arr[i] is better than current worst, replace.
         *   - Returns new heap size.
         */
        hsize = rankidx_maxheap_push_topk(
            heap_idx, hsize, k, i, rank_arr
        );
    }

    /*
     * hsize should now be k (unless cand_count < k, in which case hsize
     * == cand_count). We Assert that for sanity.
     */
    Assert(hsize == k);

    /* --------------------------------------------------------------------
     * Phase 3: determinize output order.
     *
     * The heap only tells us *which* k survived, not a stable ordering.
     * We now copy those indices into an array and sort them by:
     *     (max_penalty ASC, Path* ASC)
     *
     * We use insertion sort because k is typically small.
     * -------------------------------------------------------------------- */
    winners = (int *) palloc(sizeof(int) * k);
    for (int i = 0; i < k; i++)
        winners[i] = heap_idx[i];

    for (int i = 1; i < k; i++) {
        int wi = winners[i];
        double ppen = rank_arr[wi].max_penalty;
        Path *ppth = rank_arr[wi].path;
        int j = i - 1;

        while (j >= 0) {
            int wj = winners[j];
            double qpen = rank_arr[wj].max_penalty;
            Path *qpth = rank_arr[wj].path;

            /* compare (penalty ASC, then pointer ASC) */
            if (ppen < qpen ||
                (ppen == qpen && ppth < qpth)) {
                winners[j + 1] = winners[j];
                j--;
            } else {
                break;
            }
        }
        winners[j + 1] = wi;
    }

    /* --------------------------------------------------------------------
     * Phase 4: Rebuild the candidate list to contain ONLY the winners,
     *          in deterministic order.
     *
     * We create a brand-new List, append in sorted order,
     * and free the old List nodes.
     *
     * IMPORTANT:
     *   We do NOT free the Path objects themselves.
     * -------------------------------------------------------------------- */
    {
        List *new_list = NIL;

        for (int i = 0; i < k; i++) {
            Path *keep = rank_arr[winners[i]].path;
            new_list = lappend(new_list, keep);
        }

        list_free(candlist);
        candlist = new_list;
    }

    /* --------------------------------------------------------------------
     * Phase 5: Write back the new trimmed list into the RelOptInfo,
     *          and free temporaries.
     * -------------------------------------------------------------------- */
    if (is_partial)
        joinrel->partial_pathlist = candlist;
    else
        joinrel->pathlist = candlist;

    pfree(rank_arr);
    pfree(heap_idx);
    pfree(winners);
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
    bool is_partial
) {
    List *candlist;
    ListCell *lc;
    double *min_global;
    Sample *ss;

    /* Sanity check */
    Assert(sample_count >= 0 && sample_count <= DIST_MAX_SAMPLE);

    /*
     * Phase 0:
     * Compute per-sample global minima across ALL candidate paths.
     * Initialize an array of length sample_count to DBL_MAX,
     * then take the minimum observed value for each sample index.
     */
    min_global = (double *) palloc(sizeof(double) * sample_count);
    for (int j = 0; j < sample_count; j++)
        min_global[j] = DBL_MAX;

    /* Choose which path list to inspect: partial or full */
    candlist = is_partial ? joinrel->partial_pathlist : joinrel->pathlist;

    foreach(lc, candlist) {
        Path *p = (Path *) lfirst(lc);
        const Sample *ts = p->total_cost_sample;
        int effective;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 &&
            ts->sample_count <= DIST_MAX_SAMPLE);

        /* Only compare up to the smaller of ts->sample_count and sample_count */
        effective = Min(ts->sample_count, sample_count);

        for (int j = 0; j < effective; j++) {
            const double v = ts->sample[j];

            if (v < min_global[j])
                min_global[j] = v;
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
    if (is_partial) {
        if (joinrel->score_sample_partial == NULL)
            joinrel->score_sample_partial = (Sample *) palloc(sizeof(Sample));
        ss = joinrel->score_sample_partial;
    } else {
        if (joinrel->score_sample == NULL)
            joinrel->score_sample = (Sample *) palloc(sizeof(Sample));
        ss = joinrel->score_sample;
    }

    ss->sample_count = sample_count;

    for (int j = 0; j < sample_count; j++)
        ss->sample[j] = min_global[j];

    /*
     * Cleanup:
     * min_global was just a scratch buffer in the current memory context.
     * pfree() is not strictly required in planner code if the context
     * is short-lived, but it's good hygiene.
     */
    pfree(min_global);
}
