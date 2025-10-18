//
// Created by Xuan Chen on 2025/10/18.
//

/*
 * Per-type pruning of paths (array-based, no repalloc):
 *   - Discover distinct pathtypes (first pass) into a fixed array (upper bound = path_count).
 *   - Allocate per-type per-sample minima as a flat 2D array [groups_count][sample_count].
 *   - Compute minima within EACH pathtype.
 *   - Build a flat PathRank array and compute each path's max penalty vs. its pathtype minima.
 *   - For EACH pathtype, select winners:
 *       * up to mp_path_limit (ascending max_penalty), and
 *       * up to mc_path_limit (ascending total_cost).
 *     This is done by sorting index slices of a flat membership array (no pointer vectors).
 *   - Take the union, free non-kept paths (except IndexPath), rebuild a single list,
 *     and sort final pathlist by total_cost ascending.
 */

#include "optimizer/prunepath.h"
#include <float.h>

/* Forward Declarations (as in your file) */
typedef struct Sample Sample;
typedef struct ErrorProfileRaw ErrorProfileRaw;
typedef struct ErrorSampleParams ErrorSampleParams;
typedef struct ErrorProfile ErrorProfile;

typedef struct PathRank {
    Path *path;
    double max_penalty; /* computed vs. per-type minima */
    double total_cost;
    bool keep_mp; /* selected in top mp_path_limit by penalty within its pathtype */
    bool keep_mc; /* selected in top mc_path_limit by cost within its pathtype */
} PathRank;

/* FINAL list sort comparator for Path* by total_cost ascending */
static int
compare_path_total_cost_ptr_asc(const void *a, const void *b) {
    Path *const *pa = (Path *const *) a;
    Path *const *pb = (Path *const *) b;

    if ((*pa)->total_cost < (*pb)->total_cost) return -1;
    if ((*pa)->total_cost > (*pb)->total_cost) return 1;

    /* deterministic tiebreaker on pointer */
    if (*pa < *pb) return -1;
    if (*pa > *pb) return 1;
    return 0;
}

/* Comparators on PathRank indices (so we can sort index slices without moving structs) */
static int
compare_rank_idx_by_penalty_asc(const void *a, const void *b, void *ctx) {
    const PathRank *arr = (const PathRank *) ctx;
    int ia = *(const int *) a;
    int ib = *(const int *) b;

    const PathRank *pa = &arr[ia];
    const PathRank *pb = &arr[ib];

    if (pa->max_penalty < pb->max_penalty) return -1;
    if (pa->max_penalty > pb->max_penalty) return 1;

    /* tiebreak by pointer for determinism */
    if (pa->path < pb->path) return -1;
    if (pa->path > pb->path) return 1;
    return 0;
}

static int
compare_rank_idx_by_cost_asc(const void *a, const void *b, void *ctx) {
    const PathRank *arr = (const PathRank *) ctx;
    int ia = *(const int *) a;
    int ib = *(const int *) b;

    const PathRank *pa = &arr[ia];
    const PathRank *pb = &arr[ib];

    if (pa->total_cost < pb->total_cost) return -1;
    if (pa->total_cost > pb->total_cost) return 1;

    /* tiebreak by pointer for determinism */
    if (pa->path < pb->path) return -1;
    if (pa->path > pb->path) return 1;
    return 0;
}

/* Wrapper to use qsort_r-like API if available; otherwise emulate with a static context (PG has qsort_arg) */
#ifndef qsort_r
/* PostgreSQL provides qsort_arg in src/port/qsort.c; prefer it if available in your tree */
#define USE_QSORT_ARG
#endif

#ifdef USE_QSORT_ARG
static int
compare_rank_idx_by_penalty_asc_arg(const void *a, const void *b, void *arg) {
    return compare_rank_idx_by_penalty_asc(a, b, arg);
}

static int
compare_rank_idx_by_cost_asc_arg(const void *a, const void *b, void *arg) {
    return compare_rank_idx_by_cost_asc(a, b, arg);
}
#endif

/* Linear search in the pathtype array [0..groups_count) */
static inline int
find_type_index(const int *types, int groups_count, int pathtype) {
    for (int i = 0; i < groups_count; i++)
        if (types[i] == pathtype)
            return i;
    return -1;
}

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

        if (find_type_index(types, groups_count, pt) < 0) {
            types[groups_count++] = pt;
        }
    }

    /* Allocate per-type minima as a flat 2D array: [groups_count][sample_count] */
    double *min_flat = (double *) palloc(sizeof(double) * groups_count * sample_count);
    for (int g = 0; g < groups_count; g++) {
        for (j = 0; j < sample_count; j++) {
            min_flat[g * sample_count + j] = DBL_MAX;
        }
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
        for (j = 0; j < effective; j++) {
            const double v = ts->sample[j];
            double *base = &min_flat[gidx * sample_count];
            if (v < base[j])
                base[j] = v;
        }
    }

    /* -----------------------------
     * Phase 2: build PathRank array and compute penalties vs per-type minima
     * ----------------------------- */
    PathRank *arr = (PathRank *) palloc(sizeof(PathRank) * path_count);
    int idx = 0;

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
        arr[idx].keep_mp = false;
        arr[idx].keep_mc = false;
        idx++;
    }
    Assert(idx == path_count);

    /* -----------------------------
     * Phase 3: build per-type membership (flat) without repalloc
     *
     * We do two passes:
     *   (a) count members per type -> counts[g]
     *   (b) prefix sum -> offsets[g], fill members_flat[] with indices into arr[]
     * ----------------------------- */
    int *counts = (int *) palloc(sizeof(int) * groups_count);
    int *offsets = (int *) palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++) counts[g] = 0;

    for (int i = 0; i < path_count; i++) {
        const int gidx = find_type_index(types, groups_count, arr[i].path->pathtype);
        counts[gidx]++;
    }

    int total_members = 0;
    for (int g = 0; g < groups_count; g++) {
        offsets[g] = total_members;
        total_members += counts[g];
    }
    Assert(total_members == path_count);

    int *members_flat = (int *) palloc(sizeof(int) * path_count);
    /* temp counters reused as write heads */
    int *heads = (int *) palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++) heads[g] = offsets[g];

    for (int i = 0; i < path_count; i++) {
        const int gidx = find_type_index(types, groups_count, arr[i].path->pathtype);
        members_flat[heads[gidx]++] = i; /* store PathRank index */
    }

    /* Temporary buffer to sort a group slice by different keys without realloc */
    int max_group_size = 0;
    for (int g = 0; g < groups_count; g++)
        if (counts[g] > max_group_size) max_group_size = counts[g];
    int *buf = (int *) palloc(sizeof(int) * max_group_size);

    /* -----------------------------
     * Phase 4: per-type selections (min-penalty and min-cost) using index-slice sorts
     * ----------------------------- */
    for (int g = 0; g < groups_count; g++) {
        const int start = offsets[g];
        const int n = counts[g];
        if (n == 0)
            continue;

        /* Copy group indices into buf */
        for (int t = 0; t < n; t++)
            buf[t] = members_flat[start + t];

        /* (a) min-penalty */
        qsort_arg(buf, n, sizeof(int), compare_rank_idx_by_penalty_asc_arg, (void *) arr);

        const int take_mp = Min(mp_path_limit, n);
        for (int t = 0; t < take_mp; t++)
            arr[buf[t]].keep_mp = true;

        /* (b) min-cost */
        /* Re-copy (to avoid stable-sort dependency) */
        for (int t = 0; t < n; t++)
            buf[t] = members_flat[start + t];

#ifdef USE_QSORT_ARG
        qsort_arg(buf, n, sizeof(int), compare_rank_idx_by_cost_asc_arg, (void *) arr);
#else
        qsort_r(buf, n, sizeof(int), compare_rank_idx_by_cost_asc, (void *) arr);
#endif
        const int take_mc = Min(mc_path_limit, n);
        for (int t = 0; t < take_mc; t++)
            arr[buf[t]].keep_mc = true;

        /* Optional debug: log top few by cost for this type */
        {
            const int show = Min(n, 4);
            for (int t = 0; t < show; t++) {
                int ri = buf[t];
                elog(LOG,
                     "prune_path(type=%d): cost-rank %d: path=%p max_penalty=%.6f total_cost=%.6f",
                     types[g], t, (void*)arr[ri].path, arr[ri].max_penalty, arr[ri].total_cost);
            }
        }
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

    /* Sort final kept paths by total cost ascending */
    qsort(kept_paths, kept, sizeof(Path*), compare_path_total_cost_ptr_asc);

    /* Replace list with sorted kept paths */
    list_free(pathlist);
    List *new_list = NIL;
    for (int i = 0; i < kept; i++)
        new_list = lappend(new_list, kept_paths[i]);
    *pathlist_ptr = new_list;

    elog(LOG, "prune_path(per-type, arrays): kept %d of %d paths across %d pathtypes (mc_limit=%d, mp_limit=%d)",
         kept, path_count, groups_count, mc_path_limit, mp_path_limit);

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
    pfree(buf);
}
