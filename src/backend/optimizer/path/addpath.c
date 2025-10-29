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

/* ----------------------------------------------------------------------------
 * reconsider_pathlist
 * --------------------------------------------------------------------------*/
/*
 * reconsider_pathlist
 *
 * Replace *pathlist_ptr with at most mp_path_limit paths chosen from the
 * current *pathlist_ptr. We rank each Path by its "penalty":
 *
 *   penalty(p) = max_j ( sample_p[j] - min_global[j] )
 *
 * where min_global[j] is the minimum cost among all candidate paths at
 * sample j.  We only consider up to sample_count samples.
 *
 * Then we keep the mp_path_limit paths with the SMALLEST penalties.
 *
 * After selection we rebuild *pathlist_ptr to contain ONLY those winners.
 * For determinism we sort the winners by (penalty ASC, Path* ASC).
 */

void
reconsider_pathlist(
    PlannerInfo *root,
    int lev_index,
    int rel_index,
    int sample_count,
    int mp_path_limit
) {
    List *candlist = ((RelOptInfo *) list_nth(root->join_rel_level[lev_index], rel_index))->pathlist;
    ListCell *lc;

    /* Basic sanity */
    Assert(sample_count >= 1);
    Assert(mp_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    const int cand_count = list_length(candlist);
    if (cand_count <= 0)
        return; /* nothing to do */

    /* If we already have <= mp_path_limit, we can early-exit after
     * optionally doing a deterministic reorder if desired. But let's
     * just run the full logic for simplicity.
     */

    /* --------------------------------------------------------------------
     * Phase 0: compute per-sample global minima across ALL candidate paths
     * -------------------------------------------------------------------- */
    double *min_global = (double *) palloc(sizeof(double) * sample_count);
    for (int j = 0; j < sample_count; j++)
        min_global[j] = DBL_MAX;

    foreach(lc, candlist) {
        Path *p = (Path *) lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 && ts->sample_count <= DIST_MAX_SAMPLE);

        const int effective = Min(ts->sample_count, sample_count);

        for (int j = 0; j < effective; j++) {
            const double v = ts->sample[j];
            if (v < min_global[j])
                min_global[j] = v;
        }
    }

    /* --------------------------------------------------------------------
     * Phase 1: build PathRank array and compute penalties vs min_global
     * -------------------------------------------------------------------- */
    PathRank *rank_arr = (PathRank *) palloc(sizeof(PathRank) * cand_count);

    int idx = 0;
    foreach(lc, candlist) {
        Path *p = (Path *) lfirst(lc);
        const Sample *ts = p->total_cost_sample;
        const int effective = Min(ts->sample_count, sample_count);

        double max_pen = -DBL_MAX;

        if (effective <= 0) {
            /* Path has no samples -> make it very bad so it loses. */
            max_pen = DBL_MAX;
        } else {
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
     * Phase 2: select the global top-k (smallest penalty) using fixed-size
     * MAX-heap of indices. The heap keeps the best k so far; its root is
     * currently the *worst* among the kept (largest penalty).
     * -------------------------------------------------------------------- */
    const int k = Min(mp_path_limit, cand_count);

    int *heap_idx = (int *) palloc(sizeof(int) * Max(1, k));
    int hsize = 0;

    for (int i = 0; i < cand_count; i++) {
        hsize = rankidx_maxheap_push_topk(heap_idx, hsize, k, i, rank_arr);
    }
    /* hsize should be k (unless cand_count < k) */
    Assert(hsize == k);

    /* --------------------------------------------------------------------
     * Phase 3: materialize winners.
     *
     * heap_idx currently represents an (unordered) set of k "best" indices,
     * but not sorted ascending by penalty. We want deterministic output.
     *
     * We'll copy those indices into an array, then sort that array by:
     *   (max_penalty ASC, Path* ASC)
     * and then build the final List in that order.
     * -------------------------------------------------------------------- */

    /* Gather winner indices */
    int *winners = (int *) palloc(sizeof(int) * k);
    for (int i = 0; i < k; i++)
        winners[i] = heap_idx[i];

    /* Simple insertion sort because k is typically small. */
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

    /* Rebuild new pathlist from winners only */
    {
        List *new_list = NIL;

        for (int i = 0; i < k; i++) {
            Path *keep = rank_arr[winners[i]].path;
            new_list = lappend(new_list, keep);
        }

        /* free old list structure (but NOT the Path objects themselves) */
        list_free(candlist);
        candlist = new_list;
    }

    /* --------------------------------------------------------------------
     * Phase 4: assign result + free temporaries
     * -------------------------------------------------------------------- */
    ((RelOptInfo *) list_nth(root->join_rel_level[lev_index], rel_index))->pathlist = candlist;

    pfree(min_global);
    pfree(rank_arr);
    pfree(heap_idx);
    pfree(winners);
}
