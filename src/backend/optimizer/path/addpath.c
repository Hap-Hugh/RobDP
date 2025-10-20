//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/20.
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
path_minheap_sift_up(Path **heap, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) >> 1;
        if (!path_less_total_cost(heap[idx], heap[parent]))
            break;
        Path *tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
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

static void
path_minheap_push(Path **heap, int *pn, Path *val) {
    int n = *pn;
    heap[n] = val;
    path_minheap_sift_up(heap, n);
    *pn = n + 1;
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

/* Linear search in the pathtype array [0..groups_count). Type count is small. */
static int
find_type_index(const int *types, int groups_count, int pathtype) {
    for (int i = 0; i < groups_count; i++)
        if (types[i] == pathtype)
            return i;
    return -1;
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
 * consider_additional_path
 * --------------------------------------------------------------------------*/

void
consider_additional_path(
    List **pathlist_ptr,
    List *additional_pathlist,
    int sample_count,
    int mp_path_limit
) {
    List *pathlist = *pathlist_ptr;
    ListCell *lc;

    /* Fast exit: no additional paths => do nothing at all. */
    if (additional_pathlist == NIL)
        return;

    /* Basic sanity */
    Assert(sample_count >= 1);
    Assert(mp_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    const int add_count = list_length(additional_pathlist);

    /* -----------------------------
     * Phase 0: collect distinct pathtypes from additional_pathlist
     * ----------------------------- */
    int *types = (int *) palloc(sizeof(int) * add_count);
    int groups_count = 0;

    foreach(lc, additional_pathlist) {
        Path *p = (Path *) lfirst(lc);
        const int pt = p->pathtype;

        if (find_type_index(types, groups_count, pt) < 0)
            types[groups_count++] = pt;
    }

    /* Allocate per-type minima as a flat 2D array: [groups_count][sample_count] */
    double *min_flat = (double *) palloc(sizeof(double) * groups_count * sample_count);
    for (int g = 0; g < groups_count; g++)
        for (int j = 0; j < sample_count; j++)
            min_flat[g * sample_count + j] = DBL_MAX;

    /* -----------------------------
     * Phase 1: compute per-type per-sample minima over additional_pathlist
     * ----------------------------- */
    foreach(lc, additional_pathlist) {
        Path *p = (Path *) lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 && ts->sample_count <= DIST_MAX_SAMPLE);

        const int gidx = find_type_index(types, groups_count, p->pathtype);
        Assert(gidx >= 0);

        const int effective = Min(ts->sample_count, sample_count);
        for (int j = 0; j < effective; j++) {
            const double v = ts->sample[j];
            double *base_row = &min_flat[gidx * sample_count];
            if (v < base_row[j])
                base_row[j] = v;
        }
    }

    /* -----------------------------
     * Phase 2: build PathRank array for additional_pathlist and compute penalties
     * ----------------------------- */
    PathRank *rank_arr = (PathRank *) palloc(sizeof(PathRank) * add_count);
    int idx = 0;

    foreach(lc, additional_pathlist) {
        Path *p = (Path *) lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        const int gidx = find_type_index(types, groups_count, p->pathtype);
        Assert(gidx >= 0);

        const int effective = Min(ts->sample_count, sample_count);
        double max_pen = -DBL_MAX;

        if (effective <= 0) {
            /* No samples -> give worst penalty so it loses to any sampled path. */
            max_pen = DBL_MAX;
        } else {
            const double *group_min = &min_flat[gidx * sample_count];
            for (int j = 0; j < effective; j++) {
                const double pen = ts->sample[j] - group_min[j];
                if (pen > max_pen)
                    max_pen = pen;
            }
        }

        rank_arr[idx].path = p;
        rank_arr[idx].max_penalty = max_pen;
        idx++;
    }
    Assert(idx == add_count);

    /* -----------------------------
     * Phase 3: per-type membership bookkeeping
     * ----------------------------- */
    int *counts = (int *) palloc(sizeof(int) * groups_count);
    int *offsets = (int *) palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++) counts[g] = 0;

    for (int i = 0; i < add_count; i++) {
        const int gidx = find_type_index(types, groups_count, rank_arr[i].path->pathtype);
        counts[gidx]++;
    }

    int total_members = 0;
    for (int g = 0; g < groups_count; g++) {
        offsets[g] = total_members;
        total_members += counts[g];
    }
    Assert(total_members == add_count);

    int *members_flat = (int *) palloc(sizeof(int) * add_count);
    int *heads = (int *) palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++) heads[g] = offsets[g];

    for (int i = 0; i < add_count; i++) {
        const int gidx = find_type_index(types, groups_count, rank_arr[i].path->pathtype);
        members_flat[heads[gidx]++] = i; /* store PathRank index */
    }

    /* -----------------------------
     * Phase 4: per-type selection using MAX-HEAP of size k (keep k smallest penalties)
     * ----------------------------- */
    const int k_cap = mp_path_limit;
    /* Temporary heap buffer big enough for the largest group (bounded by k) */
    int *heap_idx = (int *) palloc(sizeof(int) * Max(1, k_cap));

    for (int g = 0; g < groups_count; g++) {
        const int start = offsets[g];
        const int n = counts[g];
        if (n == 0)
            continue;

        /* Build top-k (k = min(k_cap, n)) using fixed-size max-heap */
        int k = Min(k_cap, n);
        int hsize = 0;

        for (int t = 0; t < n; t++) {
            int cand = members_flat[start + t]; /* PathRank index */
            hsize = rankidx_maxheap_push_topk(heap_idx, hsize, k, cand, rank_arr);
        }
        Assert(hsize == k);

        /* The heap currently holds the k winners but in no particular order.
           We need them in ascending penalty to append deterministically.
           Extract them by repeatedly removing the current largest into a temp array,
           then reverse to get ascending. */
        int *tmp = (int *) palloc(sizeof(int) * k);
        for (int i = 0; i < k; i++) {
            /* pop root (largest among kept) */
            int root = heap_idx[0];
            heap_idx[0] = heap_idx[hsize - 1];
            hsize--;
            if (hsize > 0)
                rankidx_maxheap_sift_down(heap_idx, hsize, 0, rank_arr);
            tmp[i] = root;
        }
        /* Now tmp[0..k-1] are from largest->smallest; append in reverse => ascending. */
        for (int i = k - 1; i >= 0; i--) {
            Path *winner = rank_arr[tmp[i]].path;
            pathlist = lappend(pathlist, winner);
        }
        pfree(tmp);
    }

    /* -----------------------------
     * Phase 5: final sort of pathlist by total_cost ASC using MIN-HEAP (heapify + O(n) rebuild)
     * ----------------------------- */
    const int final_count = list_length(pathlist);

    if (final_count > 0) {
        /* 1) Copy pointers into a contiguous array, then heapify in O(n) */
        Path **heap = (Path **) palloc(sizeof(Path *) * final_count);
        int i = 0;
        foreach(lc, pathlist)
            heap[i++] = (Path *) lfirst(lc);
        Assert(i == final_count);

        path_minheap_heapify(heap, final_count);

        /* 2) Pop all into a sorted array (ASC), then rebuild list with lcons (O(n)) */
        Path **sorted = (Path **) palloc(sizeof(Path *) * final_count);
        int hsize = final_count;
        for (i = 0; i < final_count; i++)
            sorted[i] = path_minheap_pop(heap, &hsize); /* ascending */

        list_free(pathlist);
        List *new_list = NIL;
        /* lcons is O(1); build from tail to head to keep ascending order */
        for (i = final_count - 1; i >= 0; i--)
            new_list = lcons(sorted[i], new_list);

        pathlist = new_list;

        pfree(sorted);
        pfree(heap);
    }

    *pathlist_ptr = pathlist;

    /* -----------------------------
     * Phase 6: free temporaries
     * ----------------------------- */
    pfree(types);
    pfree(min_flat);
    pfree(rank_arr);
    pfree(counts);
    pfree(offsets);
    pfree(members_flat);
    pfree(heads);
    pfree(heap_idx);
}
