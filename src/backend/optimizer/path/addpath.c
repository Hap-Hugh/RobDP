//
// Created by Xuan Chen on 2025/10/18.
//

#include "optimizer/addpath.h"
#include <float.h>

/* Forward Declarations (as in your file) */
typedef struct Sample Sample;
typedef struct ErrorProfileRaw ErrorProfileRaw;
typedef struct ErrorSampleParams ErrorSampleParams;
typedef struct ErrorProfile ErrorProfile;

/*
 * Assumed Path fields (as in your tree):
 *   int      pathtype;
 *   double   total_cost;
 *   Sample  *total_cost_sample;  // { int sample_count; double sample[DIST_MAX_SAMPLE]; }
 */

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

/* Per-candidate ranking info (only penalty is relevant here) */
typedef struct PathRank {
    Path *path;
    double max_penalty; /* vs. per-type per-sample minima */
} PathRank;

/* Comparators on PathRank indices so we sort index slices without moving structs */
static int
compare_rank_idx_by_penalty_asc(const void *a, const void *b, const void *ctx) {
    const PathRank *arr = ctx;
    const int ia = *(const int *) a;
    const int ib = *(const int *) b;

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
compare_rank_idx_by_penalty_asc_arg(const void *a, const void *b, void *arg) {
    return compare_rank_idx_by_penalty_asc(a, b, arg);
}

/* Linear search in the pathtype array [0..groups_count) */
static int
find_type_index(const int *types, int groups_count, const int pathtype) {
    for (int i = 0; i < groups_count; i++)
        if (types[i] == pathtype)
            return i;
    return -1;
}

/* Optional: guard against duplicate insert if the same Path* already exists in list */
static bool
list_contains_path_ptr(List *lst, Path *p) {
    ListCell *lc;
    foreach(lc, lst) {
        if (lfirst(lc) == p)
            return true;
    }
    return false;
}

/*
 * consider_additional_path
 *
 * From additional_pathlist, for each pathtype select up to mp_path_limit paths
 * with the smallest max-penalty (penalty computed against per-type, per-sample minima
 * derived from additional_pathlist only). Append those winners to *pathlist_ptr
 * if not already present, and sort the final list by total_cost ascending.
 *
 * Only "min-penalty" selection is performed here. No min-cost selection.
 */
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

    /* Basic sanity checks (match your style) */
    if (additional_pathlist == NIL)
        goto sort_and_finish; /* nothing to add; still sort */

    Assert(sample_count >= 1);
    Assert(mp_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    const int add_count = list_length(additional_pathlist);

    /* -----------------------------
     * Phase 0: collect distinct pathtypes from additional_pathlist
     * ----------------------------- */
    int *types = palloc(sizeof(int) * add_count);
    int groups_count = 0;

    foreach(lc, additional_pathlist) {
        Path *p = lfirst(lc);
        const int pt = p->pathtype;

        if (find_type_index(types, groups_count, pt) < 0)
            types[groups_count++] = pt;
    }

    /* Allocate per-type minima as a flat 2D array: [groups_count][sample_count] */
    double *min_flat = (double *) palloc(sizeof(double) * groups_count * sample_count);
    for (int g = 0; g < groups_count; g++)
        for (j = 0; j < sample_count; j++)
            min_flat[g * sample_count + j] = DBL_MAX;

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
        for (j = 0; j < effective; j++) {
            const double v = ts->sample[j];
            double *base = &min_flat[gidx * sample_count];
            if (v < base[j])
                base[j] = v;
        }
    }

    /* -----------------------------
     * Phase 2: build PathRank array for additional_pathlist and compute penalties
     * ----------------------------- */
    PathRank *arr = palloc(sizeof(PathRank) * add_count);
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
                if (pen > max_pen)
                    max_pen = pen;
            }
        }

        arr[idx].path = p;
        arr[idx].max_penalty = max_pen;
        idx++;
    }
    Assert(idx == add_count);

    /* -----------------------------
     * Phase 3: per-type membership (flat) without repalloc
     * ----------------------------- */
    int *counts = palloc(sizeof(int) * groups_count);
    int *offsets = palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++) counts[g] = 0;

    for (int i = 0; i < add_count; i++) {
        const int gidx = find_type_index(types, groups_count, arr[i].path->pathtype);
        counts[gidx]++;
    }

    int total_members = 0;
    for (int g = 0; g < groups_count; g++) {
        offsets[g] = total_members;
        total_members += counts[g];
    }
    Assert(total_members == add_count);

    int *members_flat = palloc(sizeof(int) * add_count);
    int *heads = palloc(sizeof(int) * groups_count);
    for (int g = 0; g < groups_count; g++) heads[g] = offsets[g];

    for (int i = 0; i < add_count; i++) {
        const int gidx = find_type_index(types, groups_count, arr[i].path->pathtype);
        members_flat[heads[gidx]++] = i; /* store PathRank index */
    }

    /* temp buffer to sort a group slice by penalty */
    int max_group_size = 0;
    for (int g = 0; g < groups_count; g++)
        if (counts[g] > max_group_size) max_group_size = counts[g];
    int *buf = palloc(sizeof(int) * max_group_size);

    /* -----------------------------
     * Phase 4: per-type selection by MIN-PENALTY ONLY
     *           - for each type, sort indices by penalty asc and take up to mp_path_limit
     * ----------------------------- */
    for (int g = 0; g < groups_count; g++) {
        const int start = offsets[g];
        const int n = counts[g];
        if (n == 0)
            continue;

        for (int t = 0; t < n; t++)
            buf[t] = members_flat[start + t];

        qsort_arg(buf, n, sizeof(int), compare_rank_idx_by_penalty_asc_arg, arr);

        const int take_mp = Min(mp_path_limit, n);
        for (int t = 0; t < take_mp; t++) {
            Path *winner = arr[buf[t]].path;

            /* Append if not already in the base pathlist (pointer identity) */
            if (!list_contains_path_ptr(pathlist, winner))
                pathlist = lappend(pathlist, winner);
        }
    }

sort_and_finish:
    /* -----------------------------
     * Phase 5: sort final pathlist by total_cost ascending
     * ----------------------------- */
    if (pathlist != NIL) {
        const int final_count = list_length(pathlist);
        Path **arrp = palloc(sizeof(Path *) * final_count);

        int k = 0;
        foreach(lc, pathlist)
            arrp[k++] = (Path *) lfirst(lc);

        qsort(arrp, final_count, sizeof(Path *), compare_path_total_cost_ptr_asc);

        /* rebuild list */
        list_free(pathlist);
        List *new_list = NIL;
        for (int i = 0; i < final_count; i++)
            new_list = lappend(new_list, arrp[i]);

        pathlist = new_list;
        pfree(arrp);
    }

    *pathlist_ptr = pathlist;

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
        pfree(buf);
    }
}
