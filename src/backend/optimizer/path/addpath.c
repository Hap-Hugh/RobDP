//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/20.
// Modified by Xuan Chen on 2025/10/29.
// Modified by Xuan Chen on 2025/10/31.
// Modified by Xuan Chen on 2025/11/1.
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
 * CoverPathRank
 *
 * Composite ranking for robust coverage selection:
 *   - cover_rank:
 *       Integral rank from robust coverage (calc_robust_coverage), where
 *       smaller = picked earlier by the greedy coverage algorithm.
 *   - mep:
 *       Minimum expected penalty, as produced by calc_expected_penalty.
 *
 * Sorting rule:
 *   - Primary key:  cover_rank (descending).
 *   - Secondary:    mep (ascending).
 */
typedef struct CoverPathRank {
    Path *path;
    int cover_rank; /* primary key: greedy coverage rank (0,1,2,...) */
    double mep; /* secondary key: minimum expected penalty */
} CoverPathRank;

/*
 * Comparator for CoverPathRank used by qsort().
 *
 * Order:
 *   1) cover_rank descending (larger = better)
 *   2) mep ascending (smaller = better)
 */
static int
compare_cover_path_rank(const void *a, const void *b) {
    const CoverPathRank *ra = a;
    const CoverPathRank *rb = b;

    if (ra->cover_rank < rb->cover_rank)
        return 1;
    if (ra->cover_rank > rb->cover_rank)
        return -1;

    /* Secondary: mep ascending (smaller = better) */
    if (ra->mep < rb->mep)
        return -1;
    if (ra->mep > rb->mep)
        return 1;

    return 0;
}

/*
 * select_path_by_retention_set
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * using multiple scoring strategies and a round-robin selection policy.
 * Winners are appended to `*kept_list_ptr` (which may already contain entries),
 * and the function returns a List* of all pruned (not kept) Paths.
 *
 * Contract (per-strategy):
 *   - Lower score = better.
 *   - Each strategy function must:
 *       * Fill rank_arr[0..cand_count-1] with {path, score} for each node in
 *         `cand_list`, in the same iteration order.
 *       * Assign DBL_MAX (or equivalent) to paths with zero effective samples.
 *   - `min_envelope` and `sample_count` should already be prepared (e.g., clamped
 *     to available dimensions) by the caller.
 *
 * Multi-strategy policy:
 *   - We build four PathRank arrays:
 *       1) calc_worst_penalty()
 *       2) calc_expected_penalty()
 *       3) calc_expected_total_cost()
 *       4) calc_expected_penalty_with_std()
 *   - Selection is done in round-robin order over these four strategies:
 *       strategy 0 -> 1 -> 2 -> 3 -> 0 -> ...
 *   - For each strategy in turn, we pick the *best* (lowest score) path
 *     not yet selected according to that strategy. Selected paths are
 *     never picked again by later strategies.
 *   - The process stops when either:
 *       * We have selected `select_path_limit` paths, or
 *       * A full round over all strategies fails to select any new path
 *         (e.g., all remaining have DBL_MAX scores).
 *
 * Score exposure:
 *   - If `should_save_score` is true, we store the score from
 *     calc_expected_total_cost() into Path->score for both kept and dropped
 *     paths. This gives a single consistent scalar per Path, even though
 *     the selection itself is multi-strategy and round-robin.
 *
 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 *
 * Notes:
 *   - Winners are NOT sorted here; order is deterministic but not guaranteed to
 *     be strictly increasing by score. If a sorted order is required, the caller
 *     should sort `*kept_list_ptr` afterward.
 *   - Existing entries in `*kept_list_ptr` are preserved; winners are appended.
 *   - This function does NOT free list cells of `cand_list` (parameter is const).
 *     If the caller needs to release list cells of `cand_list`, do it outside
 *     after the call.
 */
static List *
select_path_by_retention_set(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    const int cand_count = list_length(cand_list);

    /* --------------------------------------------------------------------
     * Phase 1: build four PathRank arrays and compute scores via strategies.
     * Each strategy sees the same cand_list in the same iteration order.
     * -------------------------------------------------------------------- */
    PathRank *rank_worst = palloc(sizeof(PathRank) * cand_count);
    PathRank *rank_exp_penalty = palloc(sizeof(PathRank) * cand_count);
    PathRank *rank_total_cost = palloc(sizeof(PathRank) * cand_count);
    PathRank *rank_exp_penalty_std = palloc(sizeof(PathRank) * cand_count);

    /* Strategy 0: worst penalty */
    calc_worst_penalty(cand_list, rank_worst, min_envelope, sample_count);

    /* Strategy 1: expected penalty */
    calc_expected_penalty(cand_list, rank_exp_penalty, min_envelope, sample_count);

    /* Strategy 2: expected total cost (also used for Path->score exposure) */
    calc_expected_total_cost(cand_list, rank_total_cost, min_envelope, sample_count);

    /* Strategy 3: expected penalty with standard deviation */
    calc_expected_penalty_with_std(cand_list, rank_exp_penalty_std, min_envelope, sample_count);

    /* Convenience array for iterating strategies in a loop. */
    PathRank *strategies[4] = {
        rank_worst,
        rank_exp_penalty,
        rank_total_cost,
        rank_exp_penalty_std
    };

    /* --------------------------------------------------------------------
     * Phase 2: round-robin top-k selection across the four strategies.
     *
     * For each strategy in turn, pick the best (lowest-score) path that:
     *   - has not been selected yet, and
     *   - has a finite score (< DBL_MAX) in that strategy.
     *
     * Once a path is selected by any strategy, it is marked and excluded
     * from further consideration by all strategies.
     * -------------------------------------------------------------------- */
    const int k_target = Min(select_path_limit, cand_count);
    /* Mark which candidate indices are already selected as winners. */
    bool *selected = palloc0(sizeof(bool) * cand_count);

    /* Store indices of winning candidates (into [0 .. winners_cnt-1]). */
    int *winners = palloc(sizeof(int) * k_target);
    int winners_cnt = 0;

    while (winners_cnt < k_target) {
        bool progress_this_round = false;

        for (int s = 0; s < 4 && winners_cnt < k_target; s++) {
            int best_idx = -1;
            double best_score = DBL_MAX;

            /*
             * Scan all candidates for this strategy, choosing the
             * lowest-score candidate not yet selected.
             *
             * Complexity note:
             *   - This is O(4 * k_target * cand_count), which is
             *     acceptable for moderate cand_count. If cand_count becomes
             *     large, consider pre-sorting per-strategy arrays instead.
             */
            for (int i = 0; i < cand_count; i++) {
                if (selected[i])
                    continue;

                const double score = strategies[s][i].score;

                if (score < best_score) {
                    best_score = score;
                    best_idx = i;
                }
            }

            /* If we found a candidate with a valid score, select it. */
            if (best_idx >= 0 && best_score < DBL_MAX) {
                selected[best_idx] = true;
                winners[winners_cnt++] = best_idx;
                progress_this_round = true;
            }
        }

        /*
         * If a full round over all strategies failed to pick any new path,
         * we stop. Remaining candidates are considered losers.
         */
        if (!progress_this_round)
            break;
    }

    /* Number of losers: all candidates that were not selected. */
    const int losers_cnt = cand_count - winners_cnt;
    int *losers = palloc(sizeof(int) * Max(0, losers_cnt));

    if (losers_cnt > 0) {
        int writer = 0;
        for (int i = 0; i < cand_count; i++) {
            if (!selected[i]) {
                losers[writer++] = i;
            }
        }
        Assert(writer == losers_cnt);
    }

    /* --------------------------------------------------------------------
     * Phase 3: append winners to kept list and build dropped list (return).
     *
     * NOTE:
     *   - For Path->score we expose calc_expected_total_cost()'s score, so
     *     that each Path has a single, comparable scalar, even though the
     *     selection was driven by multiple strategies.
     * -------------------------------------------------------------------- */
    List *kept_list = (*kept_list_ptr != NULL) ? *kept_list_ptr : NIL;

    /* Append winners (in round-robin order) */
    for (int i = 0; i < winners_cnt; i++) {
        const int idx = winners[i];
        const PathRank rank = rank_total_cost[idx]; /* canonical score source */

        Path *keep = rank.path;
        if (should_save_score) {
            /* Expose expected total cost on Path for later stages. */
            keep->score = rank.score;
        }
        kept_list = lappend(kept_list, keep);
    }

    List *dropped_list = NIL;
    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const int idx = losers[i];
            const PathRank rank = rank_total_cost[idx]; /* same canonical source */

            Path *drop = rank.path;
            if (should_save_score) {
                /* Expose expected total cost on Path for later stages. */
                drop->score = rank.score;
            }
            dropped_list = lappend(dropped_list, drop);
        }
    }

    /* Write back survivors (append result) */
    *kept_list_ptr = kept_list;

    /* --------------------------------------------------------------------
     * Phase 4: cleanup and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_worst);
    pfree(rank_exp_penalty);
    pfree(rank_total_cost);
    pfree(rank_exp_penalty_std);

    pfree(selected);
    pfree(winners);
    pfree(losers);

    return dropped_list;
}

/*
 * select_path_by_robust_coverage
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * using a combined ranking:
 *
 *   - Primary:  cover_rank descending (larger = better, more coverage).
 *   - Secondary: mep   ascending (smaller = better, lower expected penalty).
 *
 * Winners (top-k after sorting by the above rule) are appended to
 * `*kept_list_ptr` (which may already contain entries), and the function
 * returns a List* of all pruned (not kept) Paths.
 *
 * Contract:
 *   - `calc_robust_coverage` must:
 *       * Fill a PathRank array with .path and .score for each candidate.
 *       * .score is an integer-like *coverage amount* (e.g., #covered samples).
 *   - `calc_expected_penalty` must:
 *       * Fill a PathRank array with .path and .score (MEP) for each candidate.
 *       * Lower MEP = better.
 *
 * Score exposure:
 *   - If `should_save_score` is true, we store the cover rank for both kept
 *     and dropped paths.
 *
 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 *
 * Notes:
 *   - Winners are NOT sorted by Path->score; they follow the sorted combined
 *     order described above.
 *   - Existing entries in `*kept_list_ptr` are preserved; winners are appended.
 *   - This function does NOT free list cells of `cand_list` (parameter is const).
 *     If the caller needs to release list cells of `cand_list`, do it outside
 *     after the call.
 */
static List *
select_path_by_robust_coverage(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    const int cand_count = list_length(cand_list);

    /* Clamp limit to number of candidates */
    const int k = Min(select_path_limit, cand_count);

    /* --------------------------------------------------------------------
    * Phase 1: compute robust coverage rank and minimum expected penalty.
    *
    * We use two temporary PathRank arrays:
    *   - cover_arr: scores from calc_robust_coverage (0, 1, 2, ...).
    *   - mep_arr: scores from calc_expected_penalty (double).
    *
    * Then we merge them into a single CoverPathRank array.
    * -------------------------------------------------------------------- */
    PathRank *cover_arr = palloc(sizeof(PathRank) * cand_count);
    PathRank *mep_arr = palloc(sizeof(PathRank) * cand_count);

    /* Robust coverage ranking */
    calc_robust_coverage(cand_list, cover_arr, min_envelope, sample_count);

    /* Minimum expected penalty (MEP) */
    calc_expected_penalty(cand_list, mep_arr, min_envelope, sample_count);

    /* Build composite ranking array */
    CoverPathRank *rank_arr = palloc(sizeof(CoverPathRank) * cand_count);

    for (int i = 0; i < cand_count; ++i) {
        /*
         * Both calc_robust_coverage and calc_expected_penalty must iterate
         * cand_list in the same order, so their i-th entries refer to the
         * same Path*.
         */
        Path *path_cover = cover_arr[i].path;
        const Path *path_mep = mep_arr[i].path;

        /* Sanity: they should match; if not, something is inconsistent. */
        if (path_cover != path_mep) {
            elog(ERROR, "inconsistent paths when selecting paths by robust coverages");
        }
        Assert(path_cover == path_mep);

        rank_arr[i].path = path_cover;
        rank_arr[i].cover_rank = (int) cover_arr[i].score; /* score is integer-like */
        rank_arr[i].mep = mep_arr[i].score; /* minimum expected penalty */
    }

    /* Temporary arrays no longer needed */
    pfree(cover_arr);
    pfree(mep_arr);

    /* --------------------------------------------------------------------
     * Phase 2: sort by (cover_rank DESC, mep ASC) and split into winners/losers.
     * -------------------------------------------------------------------- */
    qsort(rank_arr, cand_count, sizeof(CoverPathRank), compare_cover_path_rank);

    /* Winners: first k entries; losers: remaining entries */
    List *kept_list = (*kept_list_ptr != NULL) ? *kept_list_ptr : NIL;
    List *dropped_list = NIL;

    /* Append winners */
    for (int i = 0; i < k; i++) {
        const CoverPathRank *cover_rank = &rank_arr[i];
        Path *keep = cover_rank->path;

        if (should_save_score) {
            /*
             * Expose cover rank as Path->score, so later stages can see a scalar
             * expectation-based penalty for this Path.
             */
            keep->score = cover_rank->cover_rank;
        }

        kept_list = lappend(kept_list, keep);
    }
    /* Build dropped list (if any) */
    for (int i = k; i < cand_count; i++) {
        const CoverPathRank *cover_rank = &rank_arr[i];
        Path *drop = cover_rank->path;

        if (should_save_score) {
            /* Same convention for losers: expose cover rank into Path->score. */
            drop->score = cover_rank->cover_rank;
        }

        dropped_list = lappend(dropped_list, drop);
    }

    /* Write back survivors (append result) */
    *kept_list_ptr = kept_list;

    /* --------------------------------------------------------------------
     * Phase 3: cleanup and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_arr);

    return dropped_list;
}

/*
 * select_path_by_strategy_basic
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * according to the provided scoring strategy. Winners are appended to
 * `*kept_list_ptr` (which may already contain entries), and the function
 * returns a List* of all pruned (not kept) Paths.
 *
 * Contract:
 *   - Lower score = better.
 *   - `path_strategy_func` must:
 *       * Fill `rank_arr[0..cand_count-1]` with {path, score} for each node in
 *         `cand_list`, in the same iteration order.
 *       * Assign DBL_MAX (or equivalent) to paths with zero effective samples.
 *   - `min_envelope` and `sample_count` should already be prepared (e.g., clamped
 *     to available dimensions) by the caller.
 *
 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 *
 * Notes:
 *   - Winners are NOT sorted here; order is deterministic but not guaranteed to
 *     be strictly increasing by score. If a sorted order is required, the caller
 *     should sort `*kept_list_ptr` afterward.
 *   - If `should_save_score` is true, each kept Path gets its computed score
 *     stored into `Path->score`. Losers’ scores are not modified.
 *   - Existing entries in `*kept_list_ptr` are preserved; winners are appended.
 *   - This function does NOT free `cand_list` (parameter is const). If the caller
 *     needs to release list cells of `cand_list`, do it outside after the call.
 */
static List *
select_path_by_strategy_basic(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const path_strategy path_strategy_func,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    const int cand_count = list_length(cand_list);

    /* --------------------------------------------------------------------
     * Phase 1: build PathRank array and compute scores via strategy.
     * The strategy function fills rank_arr with (path, score).
     * -------------------------------------------------------------------- */
    PathRank *rank_arr = palloc(sizeof(PathRank) * cand_count);
    path_strategy_func(cand_list, rank_arr, min_envelope, sample_count);

    /* --------------------------------------------------------------------
     * Phase 2: select global top-k (smallest score) using a fixed MAX-heap.
     * Root of heap = worst among currently kept.
     * -------------------------------------------------------------------- */
    const int k = Min(select_path_limit, cand_count);

    int *heap_idx = palloc(sizeof(int) * Max(1, k));
    int hsize = 0;

    for (int i = 0; i < cand_count; i++) {
        hsize = rank_idx_maxheap_push_topk(heap_idx, hsize, k, i, rank_arr);
    }

    /* hsize should equal k because k = min(limit, cand_count) */
    Assert(hsize == k);

    /* --------------------------------------------------------------------
     * Phase 3: materialize winners/losers (by index). Order not finalized here.
     * -------------------------------------------------------------------- */
    int *winners = palloc(sizeof(int) * k);
    for (int i = 0; i < k; i++) {
        winners[i] = heap_idx[i];
    }

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
     * Phase 4: append winners to kept list and build dropped list (return).
     * Optionally expose per-path score for kept paths.
     * -------------------------------------------------------------------- */
    List *kept_list = (*kept_list_ptr != NULL) ? *kept_list_ptr : NIL;

    /* k = cand_count - losers_cnt. */
    for (int i = 0; i < k; i++) {
        const PathRank rank = rank_arr[winners[i]];
        Path *keep = rank.path;
        if (should_save_score) {
            /* Expose computed score on Path for later stages,
             * and for the final output after the DP process. */
            keep->score = rank.score;
        }
        kept_list = lappend(kept_list, keep);
    }

    List *dropped_list = NIL;
    if (losers_cnt > 0) {
        for (int i = 0; i < losers_cnt; i++) {
            const PathRank rank = rank_arr[losers[i]];
            Path *drop = rank.path;
            if (should_save_score) {
                /* Expose computed score on Path for later stages,
                 * and for the final output after the DP process. */
                drop->score = rank.score;
            }
            dropped_list = lappend(dropped_list, drop);
        }
    }

    /* Write back survivors (append result) */
    *kept_list_ptr = kept_list;

    /* --------------------------------------------------------------------
     * Phase 5: cleanup and return pruned paths.
     * -------------------------------------------------------------------- */
    pfree(rank_arr);
    pfree(heap_idx);
    pfree(winners);
    pfree(losers);

    return dropped_list;
}

/*
 * select_path_by_strategy_dispatch
 *
 * From the given candidate list, keep at most `select_path_limit` Paths
 * according to the provided scoring strategy. Winners are appended to
 * `*kept_list_ptr` (which may already contain entries), and the function
 * returns a List* of all pruned (not kept) Paths.

 * Return:
 *   - List* of pruned Paths (those NOT kept). If nothing is pruned, returns NIL.
 */
List *
select_path_by_strategy_dispatch(
    const List *cand_list,
    List **kept_list_ptr,
    const double *min_envelope,
    const path_strategy path_strategy_func,
    const int select_path_limit,
    const int sample_count,
    const bool should_save_score
) {
    /*
     * Policy note:
     *   - If this stage is the main objective, set should_save_score = true to
     *     expose the computed score on kept Paths for later phases.
     *   - Otherwise (retain-only policy), leave scores as-is.
     */

    /* Basic sanity */
    Assert(kept_list_ptr != NULL);
    Assert(sample_count >= 1);
    Assert(select_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    /* Early exit: no candidates; keep existing kept list untouched */
    const int cand_count = list_length(cand_list);
    if (cand_count <= 0) {
        return NIL;
    }

    if (path_strategy_func == calc_robust_coverage) {
        return select_path_by_robust_coverage(
            cand_list,
            kept_list_ptr,
            min_envelope,
            select_path_limit,
            sample_count,
            should_save_score
        );
    }
    if (path_strategy_func == calc_retention_set) {
        return select_path_by_retention_set(
            cand_list,
            kept_list_ptr,
            min_envelope,
            select_path_limit,
            sample_count,
            should_save_score
        );
    }
    return select_path_by_strategy_basic(
        cand_list,
        kept_list_ptr,
        min_envelope,
        path_strategy_func,
        select_path_limit,
        sample_count,
        should_save_score
    );
}

/*
 * set_path_score
 *
 * Purpose:
 *   Compute a score for every Path in `cand_list` using the provided strategy
 *   function, and store the result into `path->score`.
 *
 * What this function DOES:
 *   - Allocates a temporary PathRank array sized to `cand_count`.
 *   - Invokes `path_strategy_func(cand_list, rank_arr, min_envelope, sample_count)`.
 *     The strategy fills rank_arr[0..cand_count-1] with (path, score) pairs
 *     that correspond to the iteration order of `cand_list`.
 *   - Writes `rank_arr[i].score` back into the associated `Path` via `path->score`.
 *   - Frees the temporary array and returns.
 *
 * What this function DOES NOT do:
 *   - It does NOT select or prune candidates.
 *   - It does NOT sort winners or losers.
 *   - It does NOT modify list topology (no cells are freed or appended).
 *
 * Contract & Assumptions:
 *   - Lower score means better.
 *   - The strategy must assign a finite score for valid candidates and should
 *     assign DBL_MAX (or equivalent “worst”) to paths with zero effective samples.
 *   - `min_envelope` and `sample_count` have already been prepared/clamped by caller.
 *   - `sample_count >= 1` and `sample_count <= DIST_MAX_SAMPLE`.
 *
 * External invariants (referenced but not changed here):
 *   - `kept_list_ptr` and `select_path_limit` are asserted for sanity only.
 *     They are not read for control-flow in this routine and are presumed
 *     to be defined in the surrounding module.
 *
 * Inputs:
 *   - cand_list:     list of Path* to be scored.
 *   - min_envelope:  baseline (per-sample) array needed by the scoring strategy.
 *   - path_strategy_func: callback that computes path scores.
 *   - sample_count:  number of samples the strategy should consider.
 *
 * Side effects:
 *   - Each Path in cand_list receives its computed score in `Path->score`.
 *
 * Return:
 *   - void
 *
 * Error handling:
 *   - Early return if cand_list is empty.
 *   - Asserts enforce basic preconditions in assertions-enabled builds.
 */
void
set_path_score(
    const List *cand_list,
    const double *min_envelope,
    const path_strategy path_strategy_func,
    const int sample_count
) {
    /* --- Sanity checks on global/contextual invariants (not used for control flow) --- */
    Assert(kept_list_ptr != NULL);
    Assert(sample_count >= 1);
    Assert(select_path_limit >= 1);
    Assert(sample_count <= DIST_MAX_SAMPLE);

    /* --- Early exit: nothing to score --- */
    const int cand_count = list_length(cand_list);
    if (cand_count <= 0) {
        return;
    }

    /* --------------------------------------------------------------------
     * Phase 1: allocate rank array & run strategy
     *
     * The strategy fills `rank_arr` with (path, score) for each element of
     * `cand_list` in the same traversal order.
     * -------------------------------------------------------------------- */
    PathRank *rank_arr = palloc(sizeof(PathRank) * cand_count);
    path_strategy_func(cand_list, rank_arr, min_envelope, sample_count);

    /* --------------------------------------------------------------------
     * Phase 2: persist scores onto Path objects
     *
     * We only write scores back; selection/pruning is not handled here.
     * -------------------------------------------------------------------- */
    for (int i = 0; i < cand_count; i++) {
        const PathRank rank = rank_arr[i];
        Path *path = rank.path;
        path->score = rank.score;
    }

    /* --- Phase 3: cleanup temporary workspace --- */
    pfree(rank_arr);
}

/*
 * calc_score_from_pathlist
 *
 * Compute the minimum total_cost for this joinrel over its candidate paths.
 * Two values are produced:
 *
 *   joinrel->min_score          = min(path->total_cost) over pathlist
 *   joinrel->partial_min_score  = min(path->total_cost) over partial_pathlist
 *
 * This captures the best (cheapest) deterministic plan choice seen so far
 * for both regular and partial paths.  No per-sample cost vectors are used
 * here; per-round sampling is handled elsewhere.
 *
 * Notes:
 * - Either list may be empty, in which case the corresponding minimum
 *   remains DBL_MAX.
 * - Caller ensures that 'round' and 'sample_count' are valid for the
 *   surrounding sampling framework; this routine uses total_cost only.
 */
void
calc_score_from_pathlist(
    RelOptInfo *joinrel
) {
    Assert(sample_count >= 0 && sample_count <= DIST_MAX_SAMPLE);
    ListCell *lc;

    double min_score = DBL_MAX;
    foreach(lc, joinrel->pathlist) {
        const Path *path = (Path *) lfirst(lc);
        min_score = Min(path->total_cost, min_score);
    }
    joinrel->min_score = min_score;

    double partial_min_score = DBL_MAX;
    foreach(lc, joinrel->partial_pathlist) {
        const Path *partial_path = (Path *) lfirst(lc);
        partial_min_score = Min(partial_path->total_cost, partial_min_score);
    }
    joinrel->partial_min_score = partial_min_score;
}

/*
 * calc_minimum_envelope
 *
 * Accumulate per-round scalar scores for each joinrel into the first snapshot.
 * Each element of saved_join_rel_levels is a List** holding per-level joinrels
 * for one sampling round. We walk levels [2..levels_needed] in lockstep and, for
 * each joinrel, store this round's scalar score into min_rel->min_score_sample.
 *
 * Notes/Assumptions:
 * - The first snapshot acts as the accumulator (updated in-place and returned).
 * - Sample storage uses an embedded fixed-size array inside `Sample`
 *   (e.g., double sample[DIST_MAX_SAMPLE];), so no extra allocation is needed
 *   for the array itself; we only palloc the `Sample` header once at round 0.
 * - List orders match across rounds so `forboth` walks 1:1; if lengths differ,
 *   `forboth` stops at the shorter list.
 * - This routine records the per-round scalar value:
 *        Min(cur_rel->min_score, cur_rel->partial_min_score)
 *   into the sample slot for `round`. Any later elementwise-min over rounds
 *   (i.e., a true "minimum envelope") is expected to be performed elsewhere.
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
        const int round = foreach_current_index(lc);
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
                RelOptInfo *min_rel = lfirst(lc2);
                if (round == 0) {
                    /*
                     * Allocate the Sample header once; the underlying
                     * `sample[]` is an embedded fixed-size array in `Sample`,
                     * so no separate allocation is required here.
                     * Caller ensures sample_count <= array capacity.
                     */
                    min_rel->min_score_sample = palloc(sizeof(Sample));
                    min_rel->min_score_sample->sample_count = sample_count;
                }
                /* Current round's scalar score for this joinrel */
                const double cur_rel_score = Min(
                    cur_rel->min_score, cur_rel->partial_min_score
                );
                /* Store into the embedded sample array at index = round */
                min_rel->min_score_sample->sample[round] = cur_rel_score;
            }
        }
    }
    /* Return the first snapshot pointer, now populated with per-round scores. */
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
