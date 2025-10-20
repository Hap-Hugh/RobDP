//
// Created by Xuan Chen on 2025/10/18.
// Optimized by Xuan Chen on 2025/10/19: binary search for pathtype, Top-K heaps, fast dedup.
//

#include "optimizer/addpath.h"
#include <float.h>

/* Forward Declarations (assumed from your code base) */
typedef struct Sample Sample;
typedef struct ErrorProfileRaw ErrorProfileRaw;
typedef struct ErrorSampleParams ErrorSampleParams;
typedef struct ErrorProfile ErrorProfile;

/* Final list comparator: total_cost asc, tie by pointer for determinism */
static int
compare_path_total_cost_ptr_asc(const void *a, const void *b) {
    Path *const *pa = (Path *const *) a;
    Path *const *pb = (Path *const *) b;

    if ((*pa)->total_cost < (*pb)->total_cost) return -1;
    if ((*pa)->total_cost > (*pb)->total_cost) return 1;
    if (*pa < *pb) return -1;
    if (*pa > *pb) return 1;
    return 0;
}

/* Pointer ascending comparator (for dedup arrays) */
static int
compare_path_ptr_asc(const void *a, const void *b) {
    const Path *const *pa = (const Path *const *) a;
    const Path *const *pb = (const Path *const *) b;
    if (*pa < *pb) return -1;
    if (*pa > *pb) return 1;
    return 0;
}

/* Int ascending comparator (for pathtype array sort) */
static int
compare_int_asc(const void *a, const void *b) {
    const int ia = *(const int *) a;
    const int ib = *(const int *) b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

/* ---------- Binary search for pathtype -> gidx (types[] MUST be sorted) ---------- */
static inline int
find_type_index(const int *types, int groups_count, int pathtype) {
    for (int i = 0; i < groups_count; i++) {
        if (types[i] == pathtype) {
            return i;
        }
    }
    return -1;
}

/* ---------- Per-candidate ranking info (cache gidx to avoid repeated lookups) ---------- */
typedef struct PathRank {
    Path *path;
    double max_penalty; /* vs. per-type per-sample minima (computed from additional_pathlist) */
    int gidx; /* cached group index */
} PathRank;

/* ---------- Small Max-Heap for per-type Top-K (by ascending penalty) ---------- */
/*
 * Max-heap semantics: the root is the current WORST among kept-K.
 * Key = max_penalty; tie-break by Path* pointer (larger pointer = worse).
 * We store PathRank indices (ri) and view keys from arr[ri].
 */
typedef struct MaxHeap {
    int *data; /* ri indices */
    int size;
    int cap;
    const PathRank *arr;
} MaxHeap;

static int
heap_is_worse_or_equal(const MaxHeap *h, int ra, int rb) {
    const double ka = h->arr[ra].max_penalty;
    const double kb = h->arr[rb].max_penalty;
    if (ka > kb) return 1;
    if (ka < kb) return 0;
    return h->arr[ra].path >= h->arr[rb].path; /* tie by pointer: larger is worse */
}

static void
heap_swap(MaxHeap *h, int i, int j) {
    int t = h->data[i];
    h->data[i] = h->data[j];
    h->data[j] = t;
}

static void
heap_sift_up(MaxHeap *h, int i) {
    while (i > 0) {
        int parent = (i - 1) >> 1;
        if (heap_is_worse_or_equal(h, h->data[parent], h->data[i]))
            break; /* parent >= child for "worse": heap property holds */
        heap_swap(h, parent, i);
        i = parent;
    }
}

static void
heap_sift_down(MaxHeap *h, int i) {
    for (;;) {
        int left = (i << 1) + 1;
        int right = left + 1;
        int largest = i;

        if (left < h->size && !heap_is_worse_or_equal(h, h->data[largest], h->data[left]))
            largest = left;
        if (right < h->size && !heap_is_worse_or_equal(h, h->data[largest], h->data[right]))
            largest = right;
        if (largest == i)
            break;
        heap_swap(h, i, largest);
        i = largest;
    }
}

static void
heap_init(MaxHeap *h, int cap, const PathRank *arr) {
    h->cap = cap;
    h->size = 0;
    h->arr = arr;
    h->data = (cap > 0) ? (int *) palloc(sizeof(int) * cap) : NULL;
}

static void
heap_free(MaxHeap *h) {
    if (h->data) pfree(h->data);
    h->data = NULL;
    h->size = h->cap = 0;
}

static void
heap_push(MaxHeap *h, int ri) {
    h->data[h->size++] = ri;
    heap_sift_up(h, h->size - 1);
}

static void
heap_maybe_push(MaxHeap *h, int ri) {
    if (h->cap == 0) return;
    if (h->size < h->cap) {
        heap_push(h, ri);
        return;
    }
    int root = h->data[0];
    /* Better means strictly smaller penalty, or equal penalty but smaller pointer */
    const double knew = h->arr[ri].max_penalty;
    const double kroot = h->arr[root].max_penalty;
    bool better = false;
    if (knew < kroot) better = true;
    else if (knew == kroot && h->arr[ri].path < h->arr[root].path) better = true;

    if (better) {
        h->data[0] = ri;
        heap_sift_down(h, 0);
    }
}

/* ---------- Main: consider_additional_path (optimized) ---------- */
void
consider_additional_path(
    List **pathlist_ptr,
    List *additional_pathlist,
    int sample_count,
    int mp_path_limit
) {
    List *pathlist = *pathlist_ptr;
    ListCell *lc;
    int j;

    /* Nothing to add: still ensure final list is cost-sorted. */
    if (additional_pathlist == NIL)
        goto sort_and_finish;

    Assert(sample_count >= 1);
    Assert(mp_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    const int add_count = list_length(additional_pathlist);

    /* -----------------------------
     * Phase 0: collect and SORT distinct pathtypes from additional_pathlist
     * ----------------------------- */
    int *types = (int *) palloc(sizeof(int) * add_count);
    int groups_count = 0;

    foreach(lc, additional_pathlist) {
        Path *p = lfirst(lc);
        const int pt = p->pathtype;

        bool seen = false;
        for (int i = 0; i < groups_count; i++) {
            if (types[i] == pt) {
                seen = true;
                break;
            }
        }
        if (!seen) types[groups_count++] = pt;
    }
    if (groups_count > 1)
        qsort(types, groups_count, sizeof(int), compare_int_asc); /* now bsearch-able */

    /* Allocate per-type minima as a flat 2D array: [groups_count][sample_count] */
    double *min_flat = (double *) palloc(sizeof(double) * groups_count * sample_count);
    for (int g = 0; g < groups_count; g++) {
        double *base = &min_flat[g * sample_count];
        for (j = 0; j < sample_count; j++)
            base[j] = DBL_MAX;
    }

    /* -----------------------------
     * Phase 1: compute per-type per-sample minima from additional_pathlist
     * ----------------------------- */
    foreach(lc, additional_pathlist) {
        Path *p = lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 && ts->sample_count <= DIST_MAX_SAMPLE);

        const int gidx = find_type_index(types, groups_count, p->pathtype);
        Assert(gidx >= 0);

        const int effective = Min(ts->sample_count, sample_count);
        double *base = &min_flat[gidx * sample_count];
        for (j = 0; j < effective; j++) {
            const double v = ts->sample[j];
            if (v < base[j]) base[j] = v;
        }
    }

    /* -----------------------------
     * Phase 2: build PathRank array for additional_pathlist and compute penalties
     *          (cache gidx; no more lookups later)
     * ----------------------------- */
    PathRank *arr = (PathRank *) palloc(sizeof(PathRank) * add_count);
    int idx = 0;

    foreach(lc, additional_pathlist) {
        Path *p = lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        const int gidx = find_type_index(types, groups_count, p->pathtype);
        Assert(gidx >= 0);

        const int effective = Min(ts->sample_count, sample_count);
        double max_pen = -DBL_MAX;

        if (effective <= 0) {
            /* No samples -> give worst penalty so it loses to any sampled path */
            max_pen = DBL_MAX;
        } else {
            const double *base = &min_flat[gidx * sample_count];
            for (j = 0; j < effective; j++) {
                const double pen = ts->sample[j] - base[j];
                if (pen > max_pen) max_pen = pen;
            }
        }

        arr[idx].path = p;
        arr[idx].max_penalty = max_pen;
        arr[idx].gidx = gidx;
        idx++;
    }
    Assert(idx == add_count);

    /* -----------------------------
     * Phase 3: build per-type membership (counts/offsets with cached gidx)
     * ----------------------------- */
    int *counts = (int *) palloc0(sizeof(int) * groups_count);
    int *offsets = (int *) palloc0(sizeof(int) * groups_count);

    for (int i = 0; i < add_count; i++)
        counts[arr[i].gidx]++;

    int total_members = 0;
    for (int g = 0; g < groups_count; g++) {
        offsets[g] = total_members;
        total_members += counts[g];
    }
    Assert(total_members == add_count);

    int *members_flat = (int *) palloc(sizeof(int) * add_count);
    int *heads = (int *) palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++)
        heads[g] = offsets[g];

    for (int i = 0; i < add_count; i++) {
        const int gidx = arr[i].gidx;
        members_flat[heads[gidx]++] = i; /* store PathRank index */
    }

    /* -----------------------------
     * Phase 4: per-type selection by MIN-PENALTY using Top-K max-heaps
     *          Collect winners' Path* into a temporary array.
     * ----------------------------- */
    int winners_cap = (mp_path_limit > 0) ? groups_count * mp_path_limit : 0;
    if (winners_cap > add_count) winners_cap = add_count;
    Path **winners = (Path **) palloc(sizeof(Path *) * winners_cap);
    int winners_sz = 0;

    for (int g = 0; g < groups_count; g++) {
        const int start = offsets[g];
        const int n = counts[g];
        if (n == 0) continue;

        MaxHeap heap;
        heap_init(&heap, mp_path_limit, arr);

        for (int t = 0; t < n; t++) {
            int ri = members_flat[start + t];
            heap_maybe_push(&heap, ri);
        }

        /* Emit this group's winners (heap contains up to K worst-to-best unsorted) */
        for (int t = 0; t < heap.size; t++) {
            winners[winners_sz++] = arr[heap.data[t]].path;
        }
        heap_free(&heap);
    }

    /* Dedup winners themselves (by pointer)  */
    if (winners_sz > 1) {
        qsort(winners, winners_sz, sizeof(Path *), compare_path_ptr_asc);
        int u = 1;
        for (int i = 1; i < winners_sz; i++) {
            if (winners[i] != winners[u - 1])
                winners[u++] = winners[i];
        }
        winners_sz = u;
    }

    /* -----------------------------
     * Phase 4.5: fast dedup against base pathlist using binary search on pointers
     * ----------------------------- */
    int base_count = list_length(pathlist);
    Path **base_ptrs = (Path **) palloc(sizeof(Path *) * base_count);

    if (base_count > 0) {
        int k = 0;
        foreach(lc, pathlist)
            base_ptrs[k++] = (Path *) lfirst(lc);
        if (base_count > 1)
            qsort(base_ptrs, base_count, sizeof(Path *), compare_path_ptr_asc);
    }

    Path **added = (Path **) palloc(sizeof(Path *) * winners_sz);
    int added_sz = 0;

    for (int i = 0; i < winners_sz; i++) {
        Path **found = NULL;
        if (base_count > 0) {
            found = (Path **) bsearch(&winners[i],
                                      base_ptrs, base_count,
                                      sizeof(Path *), compare_path_ptr_asc);
        }
        if (found == NULL) {
            added[added_sz++] = winners[i];
        }
    }

    /* -----------------------------
     * Phase 5: build final array = base (original) + added (new), sort by total_cost
     * ----------------------------- */
    /* -----------------------------
     * Phase 5: build final array = base (original) + added (new), sort by total_cost
     * ----------------------------- */
sort_and_finish: {
        const int base_count2 = list_length(pathlist);
        int added_sz2 = 0;
        Path **added2 = NULL;

        if (additional_pathlist != NIL) {
            added_sz2 = added_sz;
            added2 = added;
        }

        const int final_count = base_count2 + added_sz2;

        const int final_cap = Max(final_count, 1);
        Path **final_arr = (Path **) palloc(sizeof(Path *) * final_cap);

        /* Copy base */
        int k = 0;
        if (base_count2 > 0) {
            foreach(lc, pathlist)
                final_arr[k++] = (Path *) lfirst(lc);
        }
        if (added_sz2 > 0) {
            for (int i = 0; i < added_sz2; i++)
                final_arr[k++] = added2[i];
        }

        /* Sort by total_cost (deterministic) and rebuild list */
        if (final_count > 1)
            qsort(final_arr, final_count, sizeof(Path *), compare_path_total_cost_ptr_asc);

        list_free(pathlist);
        List *new_list = NIL;
        for (int i = 0; i < final_count; i++)
            new_list = lappend(new_list, final_arr[i]);

        *pathlist_ptr = new_list;

        /* Free scratch for this block */
        pfree(final_arr);
        if (added_sz2 > 0) pfree(added2);
    }


    /* -----------------------------
     * Phase 6: free temporaries
     * ----------------------------- */
    if (additional_pathlist != NIL) {
        pfree(types);
        pfree(min_flat);
        pfree(arr);
        pfree(counts);
        pfree(offsets);
        pfree(members_flat);
        pfree(heads);
        pfree(winners);
        pfree(base_ptrs);
    }
}
