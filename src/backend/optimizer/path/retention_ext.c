//
// Created by Xuan Chen on 2025/11/18.
//

#include "postgres.h"
#include "optimizer/retention_ext.h"
#include "catalog/pg_operator.h"
#include "nodes/makefuncs.h"
#include "optimizer/geqo.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"

/*
 * prune_paths_by_bucket
 *
 * Apply a bucketing-based pruning over rel->pathlist.
 *
 * We bucket Paths by:
 *      (required_outer, pathkeys (NIL if parameterized), parallel_safe)
 *
 * For each bucket, we keep:
 *      - up to keep_total    Paths with lowest total_cost
 *      - up to keep_startup  Paths with lowest startup_cost
 *
 * All other Paths are removed from rel->pathlist and pfree'd
 * (except IndexPaths, which PostgreSQL conventionally does NOT pfree).
 *
 * Notes:
 *  - Order of remaining paths in rel->pathlist is preserved (stable),
 *    i.e. we do not re-sort anything.
 *  - Function modifies rel->pathlist in-place.
 *  - This is intended to be called AFTER all paths for the RelOptInfo
 *    have been constructed (similar to a post-add_path pruning pass).
 */
typedef struct PathBucket {
    List *key_pathkeys;
    Bitmapset *key_required_outer; /* only used for non-partial */
    bool key_parallel_safe; /* only used for non-partial */

    List *best_total_paths; /* sorted asc by total_cost */
    List *best_startup_paths; /* sorted asc by startup_cost */
} PathBucket;

/*
 * Bucket membership:
 *   - For normal paths: (pathkeys, required_outer, parallel_safe) must match.
 *   - For partial paths: only pathkeys are considered; required_outer and
 *     parallel_safe are ignored for bucketing.
 */
static bool
path_matches_bucket(Path *path, PathBucket *bucket, bool is_partial) {
    List *pathkeys = path->param_info ? NIL : path->pathkeys;

    if (compare_pathkeys(pathkeys, bucket->key_pathkeys) != PATHKEYS_EQUAL)
        return false;

    if (!is_partial) {
        Bitmapset *outer = PATH_REQ_OUTER(path);

        if (!bms_equal(outer, bucket->key_required_outer))
            return false;
        if (path->parallel_safe != bucket->key_parallel_safe)
            return false;
    }

    return true;
}

/*
 * Create a new bucket for a given Path.
 *
 * For partial paths, we only care about pathkeys; required_outer and
 * parallel_safe are left at defaults and not used for matching.
 */
static PathBucket *
make_path_bucket(Path *path, bool is_partial) {
    PathBucket *bucket = (PathBucket *) palloc0(sizeof(PathBucket));

    bucket->key_pathkeys = path->param_info ? NIL : path->pathkeys;

    if (!is_partial) {
        bucket->key_required_outer = PATH_REQ_OUTER(path);
        bucket->key_parallel_safe = path->parallel_safe;
    } else {
        bucket->key_required_outer = NULL; /* unused */
        bucket->key_parallel_safe = false; /* unused */
    }

    bucket->best_total_paths = NIL;
    bucket->best_startup_paths = NIL;

    return bucket;
}

static void
bucket_consider_total(PathBucket *bucket, Path *path, int limit) {
    ListCell *lc;
    int insert_at = 0;

    if (limit <= 0)
        return;

    /* Avoid duplicates */
    if (list_member_ptr(bucket->best_total_paths, path))
        return;

    /* If list is at limit and this path is not better than the worst, skip */
    if (list_length(bucket->best_total_paths) >= limit) {
        Path *worst = (Path *) llast(bucket->best_total_paths);
        if (path->total_cost >= worst->total_cost)
            return;
    }

    /* Insert by sorted total_cost */
    foreach(lc, bucket->best_total_paths) {
        Path *cur = (Path *) lfirst(lc);
        if (path->total_cost < cur->total_cost)
            break;
        insert_at++;
    }

    bucket->best_total_paths =
            list_insert_nth(bucket->best_total_paths, insert_at, path);

    /* Trim */
    if (list_length(bucket->best_total_paths) > limit)
        bucket->best_total_paths =
                list_delete_last(bucket->best_total_paths);
}

static void
bucket_consider_startup(PathBucket *bucket, Path *path, int limit) {
    ListCell *lc;
    int insert_at = 0;

    if (limit <= 0)
        return;

    if (list_length(bucket->best_startup_paths) >= limit) {
        Path *worst = (Path *) llast(bucket->best_startup_paths);
        if (path->startup_cost >= worst->startup_cost)
            return;
    }

    if (list_member_ptr(bucket->best_startup_paths, path))
        return;

    foreach(lc, bucket->best_startup_paths) {
        Path *cur = (Path *) lfirst(lc);
        if (path->startup_cost < cur->startup_cost)
            break;
        insert_at++;
    }

    bucket->best_startup_paths =
            list_insert_nth(bucket->best_startup_paths, insert_at, path);

    if (list_length(bucket->best_startup_paths) > limit)
        bucket->best_startup_paths =
                list_delete_last(bucket->best_startup_paths);
}

/*
 * Internal helper: prune a single pathlist in-place using bucket strategy.
 *
 * If is_partial == false:
 *    - bucket by (pathkeys, required_outer, parallel_safe)
 * If is_partial == true:
 *    - bucket by pathkeys only
 */
extern void
prune_pathlist_by_bucket(
    List **pathlist_ptr,
    const bool is_partial,
    const int keep_total,
    const int keep_startup
) {
    List *pathlist = *pathlist_ptr;
    List *buckets = NIL;
    List *winners = NIL;
    ListCell *lc;
    ListCell *lc_bucket;

    if (pathlist == NIL)
        return;

    /* First pass: assign all paths to buckets, select per-bucket winners */
    foreach(lc, pathlist) {
        Path *path = (Path *) lfirst(lc);
        PathBucket *bucket = NULL;

        /* Find bucket */
        foreach(lc_bucket, buckets) {
            PathBucket *b = (PathBucket *) lfirst(lc_bucket);

            if (path_matches_bucket(path, b, is_partial)) {
                bucket = b;
                break;
            }
        }

        /* Create new bucket if none matched */
        if (bucket == NULL) {
            bucket = make_path_bucket(path, is_partial);
            buckets = lappend(buckets, bucket);
        }

        bucket_consider_total(bucket, path, keep_total);
        bucket_consider_startup(bucket, path, keep_startup);
    }

    /* Collect all winners (unique) */
    foreach(lc_bucket, buckets) {
        PathBucket *bucket = (PathBucket *) lfirst(lc_bucket);
        ListCell *lc2;

        foreach(lc2, bucket->best_total_paths) {
            Path *p = (Path *) lfirst(lc2);
            if (!list_member_ptr(winners, p))
                winners = lappend(winners, p);
        }

        foreach(lc2, bucket->best_startup_paths) {
            Path *p = (Path *) lfirst(lc2);
            if (!list_member_ptr(winners, p))
                winners = lappend(winners, p);
        }
    }

    /* Second pass: remove all losers from *pathlist_ptr */
    ListCell *lc3;

    foreach(lc3, pathlist) {
        Path *path = (Path *) lfirst(lc3);

        if (!list_member_ptr(winners, path)) {
            /* Remove this cell from pathlist */
            pathlist = foreach_delete_current(pathlist, lc3);

            /* Free the Path node if appropriate */
            if (!IsA(path, IndexPath))
                pfree(path);
        }
    }

    *pathlist_ptr = pathlist;
}
