//
// Created by Xuan Chen on 2025/10/19.
//

#include "optimizer/prunepath.h"
#include <float.h>

/* ---- Per-path summary (no pathtype grouping) ---- */
typedef struct PathRank {
    Path *path;
    double max_penalty; /* vs global per-sample minima */
    double total_cost;
    bool keep_mp; /* selected in top mp_path_limit by penalty */
    bool keep_mc; /* selected in top mc_path_limit by total cost */
} PathRank;

/* FINAL comparator for Path* by total_cost ascending (deterministic) */
static int
compare_path_total_cost_ptr_asc(const void *a, const void *b) {
    Path *const *pa = a;
    Path *const *pb = b;

    if ((*pa)->total_cost < (*pb)->total_cost) return -1;
    if ((*pa)->total_cost > (*pb)->total_cost) return 1;
    if (*pa < *pb) return -1;
    if (*pa > *pb) return 1;
    return 0;
}

/* ---------- Small Max-Heap for global Top-K selection ---------- */
/*
 * We maintain a max-heap of "worse-is-greater" so the root is the current
 * worst among the kept-K. If a new candidate is better than root, we replace
 * root and sift-down. Keys come from arr[ri].{max_penalty|total_cost}.
 * Deterministic tiebreak: if keys equal, compare (Path*) pointer.
 */
typedef struct MaxHeap {
    int *data; /* stores PathRank indices (ri) */
    int size;
    int cap;
    const PathRank *arr; /* for getting key and path pointer */
    bool by_cost; /* true: key = total_cost; false: key = max_penalty */
} MaxHeap;

static double
heap_key(const MaxHeap *h, const int ri) {
    return h->by_cost ? h->arr[ri].total_cost : h->arr[ri].max_penalty;
}

static bool
is_a_worse_or_equal_than_b(const MaxHeap *h, const int ria, const int rib) {
    /* "Worse" = larger key; if equal, larger pointer is considered worse. */
    const double ka = heap_key(h, ria);
    const double kb = heap_key(h, rib);

    if (ka > kb) return true;
    if (ka < kb) return false;
    return h->arr[ria].path >= h->arr[rib].path;
}

static void
heap_swap(MaxHeap *h, const int i, const int j) {
    const int t = h->data[i];
    h->data[i] = h->data[j];
    h->data[j] = t;
}

static void
heap_sift_up(MaxHeap *h, int i) {
    while (i > 0) {
        const int parent = (i - 1) >> 1;
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
heap_init(MaxHeap *h, int cap, const PathRank *arr, bool by_cost) {
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

    /* root: current worst among kept-K (max-heap). Replace only if new is better. */
    int root = h->data[0];

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

    /* Step 1: compute GLOBAL per-sample minima (no pathtype grouping) */
    double *min_total_cost = palloc(sizeof(double) * sample_count);
    for (j = 0; j < sample_count; j++)
        min_total_cost[j] = DBL_MAX;

    foreach(lc, pathlist) {
        Path *p = lfirst(lc);
        const Sample *ts = p->total_cost_sample;

        Assert(ts != NULL);
        Assert(ts->sample_count >= 0 && ts->sample_count <= DIST_MAX_SAMPLE);

        p->should_keep = false;

        const int effective = Min(ts->sample_count, sample_count);
        for (j = 0; j < effective; j++) {
            const double v = ts->sample[j];
            if (v < min_total_cost[j])
                min_total_cost[j] = v;
        }
    }

    /* Step 2: build PathRank array (max_penalty vs global minima) */
    PathRank *arr = palloc(sizeof(PathRank) * path_count);
    int i = 0;

    foreach(lc, pathlist) {
        Path *p = lfirst(lc);
        const Sample *ts = p->total_cost_sample;
        const int effective = Min(ts->sample_count, sample_count);

        double max_pen = -DBL_MAX;
        if (effective <= 0) {
            max_pen = DBL_MAX; /* worse than any sampled path */
        } else {
            for (j = 0; j < effective; j++) {
                const double pen = ts->sample[j] - min_total_cost[j];
                if (pen > max_pen)
                    max_pen = pen;
            }
        }

        arr[i].path = p;
        arr[i].max_penalty = max_pen;
        arr[i].total_cost = p->total_cost;
        arr[i].keep_mp = false;
        arr[i].keep_mc = false;
        i++;
    }

    /* Step 3: pick global Top-K by penalty and by total cost using two max-heaps */
    MaxHeap pen_heap, cost_heap;
    heap_init(&pen_heap, mp_path_limit, arr, false);
    heap_init(&cost_heap, mc_path_limit, arr, true);

    for (int ri = 0; ri < path_count; ri++) {
        heap_maybe_push(&pen_heap, ri);
        heap_maybe_push(&cost_heap, ri);
    }

    /* Mark winners */
    for (int t = 0; t < pen_heap.size; t++)
        arr[pen_heap.data[t]].keep_mp = true;

    for (int t = 0; t < cost_heap.size; t++)
        arr[cost_heap.data[t]].keep_mc = true;

    heap_free(&pen_heap);
    heap_free(&cost_heap);

    /* Step 4: union kept paths, then sort final by total_cost asc (deterministic) */
    Path **kept_paths = palloc(sizeof(Path *) * path_count);
    int kept = 0;

    for (i = 0; i < path_count; i++) {
        if (arr[i].keep_mp || arr[i].keep_mc) {
            arr[i].path->should_keep = true;
            kept_paths[kept++] = arr[i].path;
        }
    }

    qsort(kept_paths, kept, sizeof(Path *), compare_path_total_cost_ptr_asc);

    /* Step 5: free non-kept paths (except IndexPath) */
    for (i = 0; i < path_count; i++) {
        Path *p = arr[i].path;
        if (!p->should_keep && !IsA(p, IndexPath))
            pfree(p);
    }

    /* Replace list with the sorted kept paths */
    list_free(pathlist);
    List *new_list = NIL;
    for (i = 0; i < kept; i++)
        new_list = lappend(new_list, kept_paths[i]);
    *pathlist_ptr = new_list;

    elog(LOG, "prune_path(global, heaps): kept %d of %d (mc_limit=%d, mp_limit=%d)",
         kept, path_count, mc_path_limit, mp_path_limit);

    /* free scratch */
    pfree(min_total_cost);
    pfree(arr);
    pfree(kept_paths);
}
