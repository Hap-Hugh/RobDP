//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/19.
// Modified by Xuan Chen on 2025/10/20.
//

/*
 * Per-type pruning of paths (array-based, optimized: no repalloc, Top-K with heaps),
 * with an extra "bad type" rule:
 *   - Compute global best (minimum) max_penalty across ALL paths.
 *   - For EACH pathtype, compute the group's best max_penalty.
 *   - If group_best_penalty > 2.0 * global_best_penalty, mark this pathtype as "bad".
 *   - For a bad type, keep ONLY ONE path (the one with the smallest max_penalty; tie by pointer).
 *   - For a non-bad type, keep up to:
 *       * mp_path_limit by ascending max_penalty, and
 *       * mc_path_limit by ascending total_cost,
 *     then take the union.
 *   - Free non-kept paths (except IndexPath) and rebuild a single list sorted by total_cost asc.
 */

#include "optimizer/prunepath.h"
#include <float.h>

/* Forward Declarations (same as your file) */
typedef struct Sample Sample;
typedef struct ErrorProfileRaw ErrorProfileRaw;
typedef struct ErrorSampleParams ErrorSampleParams;
typedef struct ErrorProfile ErrorProfile;

/* --- PathRank: add gidx to avoid repeated pathtype lookups --- */
typedef struct PathRank {
    Path *path;
    double max_penalty; /* computed vs. per-type minima */
    double total_cost;
    int gidx; /* cached group index for this path's pathtype */
    bool keep_mp; /* selected in top mp_path_limit by penalty within its pathtype */
    bool keep_mc; /* selected in top mc_path_limit by cost within its pathtype */
} PathRank;

/* FINAL list sort comparator for Path* by total_cost ascending (deterministic) */
static int
compare_path_total_cost_ptr_asc(const void *a, const void *b) {
    Path *const *pa = a;
    Path *const *pb = b;

    if ((*pa)->total_cost < (*pb)->total_cost) return -1;
    if ((*pa)->total_cost > (*pb)->total_cost) return 1;

    /* deterministic tiebreaker on pointer */
    if (*pa < *pb) return -1;
    if (*pa > *pb) return 1;
    return 0;
}

/* Linear search in the pathtype array [0..groups_count) */
static int
find_type_index(const int *types, const int groups_count, const int pathtype) {
    for (int i = 0; i < groups_count; i++)
        if (types[i] == pathtype)
            return i;
    return -1;
}

/* ---------- Small Max-Heap for Top-K selection ---------- */
/*
 * We maintain a max-heap of "worse-is-greater" so the root is the current
 * worst among the kept-K. If a new candidate is better than root, we replace
 * root and sift-down. Keys are taken from arr[ri].{max_penalty|total_cost}.
 * Deterministic tiebreak: if keys equal, compare by (Path*) pointer.
 */
typedef struct MaxHeap {
    int *data; /* stores PathRank indices (ri) */
    int size;
    int cap;
    const PathRank *arr; /* for getting key and path pointer */
    bool by_cost; /* true: key = total_cost; false: key = max_penalty */
} MaxHeap;

static double
heap_key(MaxHeap *h, const int ri) {
    return h->by_cost ? h->arr[ri].total_cost : h->arr[ri].max_penalty;
}

static bool
is_a_worse_or_equal_than_b(MaxHeap *h, const int ria, const int rib) {
    /* "Worse" = larger key; if equal key, larger pointer is considered worse. */
    const double ka = heap_key(h, ria);
    const double kb = heap_key(h, rib);

    if (ka > kb) return true;
    if (ka < kb) return false;

    /* Equal keys -> deterministic pointer tiebreak (larger pointer = worse) */
    return h->arr[ria].path >= h->arr[rib].path;
}

static void
heap_swap(MaxHeap *h, const int i, const int j) {
    const int tmp = h->data[i];
    h->data[i] = h->data[j];
    h->data[j] = tmp;
}

static void
heap_sift_up(MaxHeap *h, int i) {
    while (i > 0) {
        const int parent = (i - 1) >> 1;
        /* If parent is already worse-or-equal than child, heap property holds */
        if (is_a_worse_or_equal_than_b(h, h->data[parent], h->data[i]))
            break;
        heap_swap(h, parent, i);
        i = parent;
    }
}

static void
heap_sift_down(MaxHeap *h, int i) {
    for (;;) {
        const int left = (i << 1) + 1;
        const int right = left + 1;
        int largest = i;

        if (left < h->size &&
            !is_a_worse_or_equal_than_b(h, h->data[largest], h->data[left]))
            largest = left;

        if (right < h->size &&
            !is_a_worse_or_equal_than_b(h, h->data[largest], h->data[right]))
            largest = right;

        if (largest == i)
            break;

        heap_swap(h, i, largest);
        i = largest;
    }
}

static void
heap_init(MaxHeap *h, const int cap, const PathRank *arr, bool by_cost) {
    h->cap = cap;
    h->size = 0;
    h->arr = arr;
    h->by_cost = by_cost;
    h->data = (cap > 0) ? (int *) palloc(sizeof(int) * cap) : NULL;
}

static void
heap_free(MaxHeap *h) {
    if (h->data)
        pfree(h->data);
    h->data = NULL;
    h->size = h->cap = 0;
}

/* push assumes size < cap */
static void
heap_push(MaxHeap *h, const int ri) {
    h->data[h->size++] = ri;
    heap_sift_up(h, h->size - 1);
}

/* Keep Top-K best (smallest key). If cap==0, do nothing. */
static void
heap_maybe_push(MaxHeap *h, const int ri) {
    if (h->cap == 0)
        return;

    if (h->size < h->cap) {
        heap_push(h, ri);
        return;
    }

    /* root is current worst among kept-K (max-heap). If new is better, replace root. */
    int root = h->data[0];

    /* "Better" means strictly smaller key, or equal key but smaller pointer. */
    const double knew = heap_key(h, ri);
    const double kroot = heap_key(h, root);

    bool better = false;
    if (knew < kroot) better = true;
    else if (knew == kroot && h->arr[ri].path < h->arr[root].path) better = true;

    if (better) {
        h->data[0] = ri;
        heap_sift_down(h, 0);
    }
}

/* --------------------------------------------------------- */

void
prune_path(
    List **pathlist_ptr, const int sample_count,
    const int mc_path_limit, const int mp_path_limit
) {
    List *pathlist = *pathlist_ptr;
    ListCell *lc;
    int j;

    Assert(pathlist != NIL);
    Assert(sample_count >= 1);
    Assert(mp_path_limit >= 1);
    Assert(mc_path_limit >= 0);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    const int path_count = list_length(pathlist);

    /* -----------------------------
     * Phase 0: collect distinct pathtypes (upper bound = path_count)
     * ----------------------------- */
    int *types = (int *) palloc(sizeof(int) * path_count);
    int groups_count = 0;

    foreach(lc, pathlist) {
        Path *p = lfirst(lc);
        const int pt = p->pathtype; /* assumed existing field */

        if (find_type_index(types, groups_count, pt) < 0)
            types[groups_count++] = pt;
    }

    /* Allocate per-type minima as a flat 2D array: [groups_count][sample_count] */
    double *min_flat = (double *) palloc(sizeof(double) * groups_count * sample_count);
    for (int g = 0; g < groups_count; g++) {
        double *base = &min_flat[g * sample_count];
        for (j = 0; j < sample_count; j++)
            base[j] = DBL_MAX;
    }

    /* -----------------------------
     * Phase 1: compute per-type per-sample minima of total_cost samples
     * ----------------------------- */
    foreach(lc, pathlist) {
        Path *p = lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 && ts->sample_count <= DIST_MAX_SAMPLE);

        p->should_keep = false;

        const int gidx = find_type_index(types, groups_count, p->pathtype);
        Assert(gidx >= 0);

        const int effective = Min(ts->sample_count, sample_count);
        double *base = &min_flat[gidx * sample_count];
        for (j = 0; j < effective; j++) {
            const double v = ts->sample[j];
            if (v < base[j])
                base[j] = v;
        }
    }

    /* -----------------------------
     * Phase 2: build PathRank array and compute penalties vs per-type minima
     *          ALSO record global best (minimum) max_penalty across ALL paths.
     * ----------------------------- */
    PathRank *arr = (PathRank *) palloc(sizeof(PathRank) * path_count);
    int idx = 0;

    double global_best_penalty = DBL_MAX;

    foreach(lc, pathlist) {
        Path *p = lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        const int gidx = find_type_index(types, groups_count, p->pathtype);
        Assert(gidx >= 0);

        const int effective = Min(ts->sample_count, sample_count);
        double max_pen = -DBL_MAX;

        if (effective <= 0) {
            /* No samples: set to worst penalty so it loses to any sampled path */
            max_pen = DBL_MAX;
        } else {
            const double *base = &min_flat[gidx * sample_count];
            for (j = 0; j < effective; j++) {
                const double pen = ts->sample[j] - base[j];
                if (pen > max_pen)
                    max_pen = pen;
            }
        }

        arr[idx].path = p;
        arr[idx].max_penalty = max_pen;
        arr[idx].total_cost = p->total_cost;
        arr[idx].gidx = gidx; /* cache pathtype group index */
        arr[idx].keep_mp = false;
        arr[idx].keep_mc = false;

        if (max_pen < global_best_penalty)
            global_best_penalty = max_pen;

        idx++;
    }
    Assert(idx == path_count);

    /* If no valid samples at all, global_best_penalty is DBL_MAX.
       In that degenerate case, we do NOT mark any type as bad. */

    /* -----------------------------
     * Phase 3: build per-type membership (flat) without repalloc
     *          (using cached gidx; no repeated find_type_index)
     * ----------------------------- */
    int *counts = (int *) palloc0(sizeof(int) * groups_count);
    int *offsets = (int *) palloc0(sizeof(int) * groups_count);

    for (int i = 0; i < path_count; i++)
        counts[arr[i].gidx]++;

    int total_members = 0;
    for (int g = 0; g < groups_count; g++) {
        offsets[g] = total_members;
        total_members += counts[g];
    }
    Assert(total_members == path_count);

    int *members_flat = (int *) palloc(sizeof(int) * path_count);
    int *heads = (int *) palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++)
        heads[g] = offsets[g];

    for (int i = 0; i < path_count; i++) {
        const int g = arr[i].gidx;
        members_flat[heads[g]++] = i; /* store PathRank index */
    }

    /* -----------------------------
     * Phase 4: per-type selection
     *          - Determine "bad type" by comparing group's best penalty with 2.0 * global best.
     *          - If bad: keep exactly ONE by penalty (smallest penalty, tie by pointer).
     *          - Else: union of Top-K by penalty and Top-K by cost (as before).
     * ----------------------------- */
    for (int g = 0; g < groups_count; g++) {
        const int start = offsets[g];
        const int n = counts[g];
        if (n == 0)
            continue;

        /* Compute group's best (minimum) max_penalty */
        double group_best_penalty = DBL_MAX;
        for (int t = 0; t < n; t++) {
            const int ri = members_flat[start + t];
            if (arr[ri].max_penalty < group_best_penalty)
                group_best_penalty = arr[ri].max_penalty;
        }

        /* Decide if this group is bad */
        bool is_bad_group = false;
        if (global_best_penalty < DBL_MAX) {
            /* only meaningful if we saw any valid sample */
            if (group_best_penalty > 2.0 * global_best_penalty)
                is_bad_group = true;
        }

        if (is_bad_group) {
            /* Keep exactly ONE by penalty: find the minimal (penalty asc, pointer asc) */
            int best_ri = -1;
            for (int t = 0; t < n; t++) {
                const int ri = members_flat[start + t];
                if (best_ri < 0) {
                    best_ri = ri;
                } else {
                    const double pa = arr[ri].max_penalty;
                    const double pb = arr[best_ri].max_penalty;
                    if (pa < pb || (pa == pb && arr[ri].path < arr[best_ri].path))
                        best_ri = ri;
                }
            }
            if (best_ri >= 0)
                arr[best_ri].keep_mp = true; /* mark via penalty channel */
            /* Do NOT select by cost to ensure the "only 1" rule. */
            continue;
        }

        /* Non-bad group: original union-of-two-heaps selection */
        MaxHeap pen_heap;
        MaxHeap cost_heap;
        heap_init(&pen_heap, mp_path_limit, arr, false);
        heap_init(&cost_heap, mc_path_limit, arr, true);

        for (int t = 0; t < n; t++) {
            const int ri = members_flat[start + t];
            heap_maybe_push(&pen_heap, ri);
            heap_maybe_push(&cost_heap, ri);
        }

        for (int t = 0; t < pen_heap.size; t++)
            arr[pen_heap.data[t]].keep_mp = true;

        for (int t = 0; t < cost_heap.size; t++)
            arr[cost_heap.data[t]].keep_mc = true;

        heap_free(&pen_heap);
        heap_free(&cost_heap);
    }

    /* -----------------------------
     * Phase 5: union, free non-kept, and rebuild final list sorted by total_cost
     * ----------------------------- */
    Path **kept_paths = (Path **) palloc(sizeof(Path *) * path_count);
    int kept = 0;
    for (int i = 0; i < path_count; i++) {
        if (arr[i].keep_mp || arr[i].keep_mc) {
            arr[i].path->should_keep = true;
            kept_paths[kept++] = arr[i].path;
        }
    }

    /* Free non-kept paths (except IndexPath) */
    for (int i = 0; i < path_count; i++) {
        Path *p = arr[i].path;
        if (!p->should_keep) {
            if (!IsA(p, IndexPath))
                pfree(p);
        }
    }

    /* Sort final kept paths by total cost ascending (deterministic tie-break) */
    if (kept > 1)
        qsort(kept_paths, kept, sizeof(Path *), compare_path_total_cost_ptr_asc);

    /* Replace list with sorted kept paths */
    list_free(pathlist);
    List *new_list = NIL;
    for (int i = 0; i < kept; i++)
        new_list = lappend(new_list, kept_paths[i]);
    *pathlist_ptr = new_list;

    elog(LOG,
         "prune_path(per-type + bad-type rule): kept %d of %d paths across %d pathtypes (mc_limit=%d, mp_limit=%d; global_best_pen=%.6g)",
         kept, path_count, groups_count, mc_path_limit, mp_path_limit, global_best_penalty);

    /* -----------------------------
     * Phase 6: free temporaries
     * ----------------------------- */
    pfree(types);
    pfree(min_flat);
    pfree(arr);
    pfree(counts);
    pfree(offsets);
    pfree(members_flat);
    pfree(heads);
    pfree(kept_paths);
}
