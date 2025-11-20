//
// Created by Xuan Chen on 2025/10/31.
// Modified by Xuan Chen on 2025/11/9.
// Modified by Xuan Chen on 2025/11/12.
//

#define PENALTY_TOLERANCE_FACTOR 1.2

#include "optimizer/pathstrategy.h"

#include <float.h>

typedef double (*basic_select_path_strategy_impl)(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_penalty_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_penalty_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_total_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_total_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_worst_penalty_with_std_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_expected_penalty_with_std_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static double calc_4th_worst_penalty_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_envelope,
    int effective
);

static void basic_select_path_strategy_helper(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count,
    basic_select_path_strategy_impl basic_select_path_strategy_impl_func
);

extern void calc_worst_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_penalty_impl
    );
}

extern void calc_expected_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_penalty_impl
    );
}

extern void calc_worst_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_total_cost_impl
    );
}

extern void calc_expected_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_total_cost_impl
    );
}

extern void calc_worst_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_startup_cost_impl
    );
}

extern void calc_expected_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_startup_cost_impl
    );
}

static void basic_select_path_strategy_helper(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count,
    const basic_select_path_strategy_impl basic_select_path_strategy_impl_func
) {
    int idx = 0;
    ListCell *lc;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);
        const Sample *startup_cost_sample = path->startup_cost_sample;
        const Sample *total_cost_sample = path->total_cost_sample;

        Assert(startup_cost_sample != NULL);
        Assert(startup_cost_sample->sample_count >= 0 &&
            startup_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        Assert(total_cost_sample != NULL);
        Assert(total_cost_sample->sample_count >= 0 &&
            total_cost_sample->sample_count <= DIST_MAX_SAMPLE);

        int effective = Min(startup_cost_sample->sample_count, total_cost_sample->sample_count);
        effective = Min(effective, sample_count);
        Assert(effective >= 0);

        double score_val;
        if (effective == 0) {
            /* No samples => automatically worst */
            score_val = DBL_MAX;
        } else {
            score_val = basic_select_path_strategy_impl_func(
                startup_cost_sample, total_cost_sample, min_envelope, effective
            );
        }

        rank_arr[idx].path = path;
        rank_arr[idx].score = score_val;
        idx++;
    }
    Assert(idx == cand_count);
}

/*
 * calc_worst_penalty_impl
 *
 * Compute the worst (maximum) penalty among samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 *
 * This metric reflects the worst-case deviation from the baseline.
 * Zero means the path is comparable to the best observed path.
 */
double calc_worst_penalty_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    /* Score >= 0 always */
    double worst_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        const double cur_penalty =
                (cur_sample > cur_thresh) ? (cur_sample - min_envelope[i]) : 0.0;

        if (cur_penalty > worst_penalty)
            worst_penalty = cur_penalty;
    }
    return worst_penalty;
}

/*
 * calc_expected_penalty_impl
 *
 * Compute the average penalty across all samples.
 *
 * This metric reflects expected (mean) deviation from the baseline,
 * instead of worst-case deviation.
 */
double calc_expected_penalty_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    double expected_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        expected_penalty +=
                (cur_sample > cur_thresh) ? (cur_sample - min_envelope[i]) : 0.0;
    }
    return expected_penalty / (double) effective;
}

/*
 * Compute standard deviation of penalty over the first `effective` samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 */
static double compute_std_penalty_sample(
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    if (effective <= 1)
        return 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);

    /* Welford's online algorithm over penalty(j) */
    double mean = 0.0;
    double M2 = 0.0;
    int n = 0;

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        const double penalty =
                (cur_sample > cur_thresh) ? (cur_sample - min_envelope[i]) : 0.0;

        const double x = penalty;
        n += 1;
        const double delta = x - mean;
        mean += delta / n;
        const double delta2 = x - mean;
        M2 += delta * delta2;
    }

    /* Population standard deviation over `effective` elements */
    const double variance = M2 / (double) effective;
    return sqrt(variance);
}

/*
 * calc_worst_penalty_with_std_impl
 *
 * Compute the worst (maximum) penalty among samples, then add the STD of the
 * penalty across the same `effective` samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 *
 * Final score:
 *   return max_j penalty(j)  +  std(penalty[0..effective-1])
 *
 * Zero penalty means the path is comparable to the best observed path; the
 * additional STD term rewards stability (lower variance).
 */
double calc_worst_penalty_with_std_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    /* Score >= 0 always (penalty >= 0, std >= 0) */
    double worst_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        const double cur_penalty =
                (cur_sample > cur_thresh) ? (cur_sample - min_envelope[i]) : 0.0;

        if (cur_penalty > worst_penalty)
            worst_penalty = cur_penalty;
    }

    /* Add STD of penalty */
    const double std_penalty =
            compute_std_penalty_sample(total_cost_sample, min_envelope, effective);
    return worst_penalty + std_penalty;
}

/*
 * calc_expected_penalty_with_std_impl
 *
 * Compute the mean penalty across samples, then add the STD of the
 * penalty across the same `effective` samples.
 *
 * Final score:
 *   return mean_j penalty(j) + std(penalty[0..effective-1])
 */
double calc_expected_penalty_with_std_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    double sum_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        sum_penalty +=
                (cur_sample > cur_thresh) ? (cur_sample - min_envelope[i]) : 0.0;
    }

    const double mean_penalty = sum_penalty / (double) effective;

    /* Add STD of penalty */
    const double std_penalty =
            compute_std_penalty_sample(total_cost_sample, min_envelope, effective);
    return mean_penalty + std_penalty;
}

/*
 * calc_4th_worst_penalty_impl
 *
 * Compute the 4th worst (4th largest) penalty among samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_envelope(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_envelope(j))
 *                  : 0.0
 *
 * We track the top 4 penalties in descending order while scanning.
 * If fewer than 4 samples exist, we return the smallest among the
 * collected penalties (i.e., the "Nth worst" where N = effective).
 */
double calc_4th_worst_penalty_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope,
    const int effective
) {
    /* Keep the top 4 penalties in descending order:
     * top4[0] = worst (largest)
     * top4[3] = 4th worst
     */
    double top4[4] = {0.0, 0.0, 0.0, 0.0};
    int seen = 0; /* number of inserted samples, capped at 4 */

    Assert(total_cost_sample != NULL);
    Assert(min_envelope != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double thresh = min_envelope[i] * PENALTY_TOLERANCE_FACTOR;
        const double val = total_cost_sample->sample[i];
        const double pen = (val > thresh) ? (val - min_envelope[i]) : 0.0;

        /* If we already have 4 penalties and this one is not better,
         * it cannot enter the top 4 list.
         */
        if (seen >= 4 && pen <= top4[3]) {
            continue;
        }

        /* Insert penalty into the descending list */
        if (pen > top4[0]) {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = top4[0];
            top4[0] = pen;
        } else if (pen > top4[1]) {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = pen;
        } else if (pen > top4[2]) {
            top4[3] = top4[2];
            top4[2] = pen;
        } else {
            /* pen goes into 4th position */
            top4[3] = pen;
        }

        if (seen < 4) {
            seen++;
        }
    }

    /* If we processed >= 4 samples, return the 4th worst.
     * Otherwise return the last valid one (i.e., "seen-th worst").
     */
    if (seen >= 4) {
        return top4[3];
    }
    return top4[seen - 1];
    /* seen >= 1 because effective > 0 */
}

/*
 * calc_worst_total_cost_impl
 *
 * Return the maximum value among total_cost samples.
 *
 * This corresponds to the worst-case total execution cost.
 */
double calc_worst_total_cost_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope, /* unused */
    const int effective
) {
    double worst_total_cost = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_sample = total_cost_sample->sample[i];
        if (cur_sample > worst_total_cost)
            worst_total_cost = cur_sample;
    }
    return worst_total_cost;
}

/*
 * calc_expected_total_cost_impl
 *
 * Return the average total cost across samples.
 *
 * This corresponds to expected execution cost.
 */
double calc_expected_total_cost_impl(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_envelope, /* unused */
    const int effective
) {
    double expected_total_cost = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i)
        expected_total_cost += total_cost_sample->sample[i];

    return expected_total_cost / (double) effective;
}

/*
 * calc_worst_startup_cost_impl
 *
 * Return the maximum startup cost among samples.
 *
 * This is the worst-case startup cost, useful for pipelining latency mode.
 */
double calc_worst_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample, /* unused */
    const double *min_envelope, /* unused */
    const int effective
) {
    double worst_startup_cost = 0.0;

    Assert(startup_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_sample = startup_cost_sample->sample[i];
        if (cur_sample > worst_startup_cost)
            worst_startup_cost = cur_sample;
    }
    return worst_startup_cost;
}

/*
 * calc_expected_startup_cost_impl
 *
 * Return the average startup cost across samples.
 *
 * This reflects expected first-tuple latency for pipelined plans.
 */
double calc_expected_startup_cost_impl(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample, /* unused */
    const double *min_envelope, /* unused */
    const int effective
) {
    double expected_startup_cost = 0.0;

    Assert(startup_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i)
        expected_startup_cost += startup_cost_sample->sample[i];

    return expected_startup_cost / (double) effective;
}

static bool covers_under_eps(
    const double path_cost,
    const double opt_cost,
    const double eps
) {
    if (opt_cost == 0.0) {
        return path_cost == 0.0; /* exact match when optimum is 0 */
    }
    return path_cost <= opt_cost * (1.0 + eps);
}

/*
 * Helper to fetch the "cost" at sample index s for a given Path.
 *
 * Assumptions per your constraints:
 *   - sample_count > 1
 *   - path->rows_sample != NULL
 *
 * Rules implemented:
 *   - If samp->sample_count == sample_count: return samp->sample[s].
 *   - If samp->sample_count == 1:           treat as constant across all s; return samp->sample[0].
 *   - Otherwise: return false (caller may choose to skip this (path,s) pair).
 *
 * Returns true on success and writes *out_cost; otherwise false.
 */
static bool fetch_cost_at(
    const Path *path,
    const int s,
    const int sample_count,
    double *out_cost
) {
    const Sample *samp = path->rows_sample;

    /* Expected by the caller */
    if (samp == NULL)
        return false;

    if (samp->sample_count == sample_count) {
        /* Per-sample vector */
        *out_cost = samp->sample[s];
        return true;
    }
    if (samp->sample_count == 1) {
        /* Constant across all s */
        *out_cost = samp->sample[0];
        return true;
    }
    /* Incompatible shape (should not happen) */
    return false;
}

/*
 * calc_robust_coverage
 *
 * Greedy robust coverage with scoring:
 *   - Compute per-sample optimum cost opt[s] across all candidates.
 *   - While uncovered samples remain:
 *       use a greedy loop to pick candidates that bring the largest *new*
 *       coverage under eps, and log each pick (gain, cumulative coverage).
 *   - After the greedy loop, compute for every candidate its *total*
 *     robust coverage (over all samples, independent of greedy order),
 *     and store that as PathRank.score.
 *
 * Inputs:
 *   - cand_list: List* of Path*; each Path has rows_sample != NULL
 *   - rank_arr : preallocated array of length nplans; we fill .path and .score
 *   - min_envelope: unused
 *   - sample_count: > 1
 *
 * Outputs (via rank_arr):
 *   - rank_arr[i].path  = i-th candidate Path
 *   - rank_arr[i].score = robust coverage count
 *                         (# of samples whose cost is within eps of per-sample optimum)
 *
 * Logging (LOG level):
 *   - After computing opt[s], we log a brief summary.
 *   - Each greedy pick logs: pick index, plan index, gain, cumulative coverage.
 *   - Final summary logs total coverage and #picks.
 */
extern void calc_robust_coverage(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope /* unused */,
    const int sample_count
) {
    const int nplans = list_length((List *) cand_list);
    const double eps = ROBUST_EPS_DEFAULT;

    (void) min_envelope; /* explicitly unused */

    if (nplans <= 0) {
        elog(LOG, "[robust_cover] No candidates.");
        return;
    }

    /* --- Materialize candidates into an indexable array (avoid repeated list_nth) --- */
    Path **paths = (Path **) palloc(sizeof(Path *) * nplans);
    {
        int i = 0;
        ListCell *lc;
        foreach(lc, (List *) cand_list)
            paths[i++] = (Path *) lfirst(lc);
    }

    /* --- Working arrays --- */
    double *opt = (double *) palloc(sizeof(double) * sample_count);
    bool *uncovered = (bool *) palloc(sizeof(bool) * sample_count);
    bool *selected = (bool *) palloc0(sizeof(bool) * nplans); /* init false */

    for (int s = 0; s < sample_count; s++) {
        opt[s] = DBL_MAX;
        uncovered[s] = true;
    }

    /* --- 1) Compute per-sample minima opt[s] across all candidates --- */
    for (int i = 0; i < nplans; i++) {
        const Path *path = paths[i];

        for (int s = 0; s < sample_count; s++) {
            double v;
            if (!fetch_cost_at(path, s, sample_count, &v))
                elog(ERROR, "[robust_cover] candidate %d has incompatible rows_sample shape.", i);

            if (v < opt[s])
                opt[s] = v;
        }
    }

    /* Log a brief statistic about opt[] for sanity (min/max over s). */
    {
        double mn = DBL_MAX, mx = -DBL_MAX;
        for (int s = 0; s < sample_count; s++) {
            if (opt[s] < mn) mn = opt[s];
            if (opt[s] > mx) mx = opt[s];
        }
        elog(LOG,
             "[robust_cover] opt computed over %d samples: min=%.6g max=%.6g",
             sample_count, mn, mx);
    }

    /* --- Initialize rank_arr with path pointers; score will be filled later --- */
    for (int i = 0; i < nplans; i++) {
        rank_arr[i].path = paths[i];
        rank_arr[i].score = 0.0; /* will be overwritten with coverage count */
    }

    /* --- 2) Greedy selection loop (for coverage set selection + logging) --- */
    int covered_total = 0; /* # of covered samples so far */
    int pick_index = 0; /* 0-based index of picks for logging */
    int picked = 0; /* # of candidates actually improving coverage */

    while (covered_total < sample_count) {
        int best_plan = -1;
        int best_gain = 0;

        /* Find candidate with maximum new coverage */
        for (int i = 0; i < nplans; i++) {
            if (selected[i])
                continue;

            const Path *path = paths[i];

            int gain = 0;
            for (int s = 0; s < sample_count; s++) {
                if (!uncovered[s])
                    continue;

                double v;
                if (!fetch_cost_at(path, s, sample_count, &v))
                    continue;

                if (covers_under_eps(v, opt[s], eps))
                    gain++;
            }

            if (gain > best_gain) {
                best_gain = gain;
                best_plan = i;
            }
        }

        if (best_plan < 0 || best_gain == 0) {
            elog(LOG, "[robust_cover] No more gains (best_gain=%d). Stopping greedy.", best_gain);
            break;
        }

        /* Commit best_plan of this round */
        {
            const Path *path = paths[best_plan];

            selected[best_plan] = true;
            picked++;
            pick_index++;

            /* mark newly covered samples */
            int newly = 0;
            for (int s = 0; s < sample_count; s++) {
                if (!uncovered[s])
                    continue;

                double v;
                if (!fetch_cost_at(path, s, sample_count, &v))
                    continue;

                if (covers_under_eps(v, opt[s], eps)) {
                    uncovered[s] = false;
                    newly++;
                }
            }
            covered_total += newly;

            const double cov_pct =
                    (sample_count > 0)
                        ? (100.0 * (double) covered_total / (double) sample_count)
                        : 0.0;

            elog(LOG,
                 "[robust_cover] Pick #%d: plan_idx=%d, gain=%d, covered=%d/%d (%.1f%%)",
                 pick_index, /* 1-based in logs */
                 best_plan, newly, covered_total, sample_count, cov_pct);
        }
    }

    /* --- 3) Compute coverage-based scores for all candidates --- */
    for (int i = 0; i < nplans; i++) {
        const Path *path = paths[i];
        int coverage = 0;

        for (int s = 0; s < sample_count; s++) {
            double v;
            if (!fetch_cost_at(path, s, sample_count, &v))
                continue; /* under your assumptions, this should not happen */

            if (covers_under_eps(v, opt[s], eps))
                coverage++;
        }

        /* score = coverage count (# covered samples) */
        rank_arr[i].score = (double) coverage;
    }

    /* Final coverage log (based on greedy result) */
    {
        int still_uncovered = 0;
        for (int s = 0; s < sample_count; s++)
            if (uncovered[s]) still_uncovered++;

        const int covered = sample_count - still_uncovered;
        const double cov_pct =
                (sample_count > 0)
                    ? (100.0 * (double) covered / (double) sample_count)
                    : 0.0;

        elog(LOG,
             "[robust_cover] Final: picked=%d/%d, covered=%d/%d (%.1f%%)",
             picked, nplans, covered, sample_count, cov_pct);
    }

    /* --- cleanup --- */
    pfree(paths);
    pfree(opt);
    pfree(uncovered);
    pfree(selected);
}

static unsigned int rng_state = 123456789u;

static unsigned int xorshift32(void) {
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static double rng_uniform01(void) {
    return xorshift32() / (double) UINT32_MAX;
}

extern void calc_random_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    int idx = 0;
    ListCell *lc;
    foreach(lc, cand_list) {
        Path *path = lfirst(lc);
        rank_arr[idx].path = path;
        rank_arr[idx].score = rng_uniform01();
        idx++;
    }
    Assert(idx == cand_count);
}

extern void calc_postgres_original_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    elog(ERROR, "`calc_postgres_original_score` not implemented.");
}

extern void calc_jointype_based_score(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    elog(ERROR, "`calc_jointype_based_score` not implemented.");
}

extern void calc_worst_penalty_with_std(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_worst_penalty_with_std_impl
    );
}

extern void calc_expected_penalty_with_std(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_expected_penalty_with_std_impl
    );
}

extern void calc_4th_worst_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    const int sample_count
) {
    basic_select_path_strategy_helper(
        cand_list, rank_arr, min_envelope, sample_count, calc_4th_worst_penalty_impl
    );
}

/* Global strategy array */
path_strategy path_strategy_funcs[13] = {
    [0] = calc_worst_penalty,
    [1] = calc_expected_penalty,
    [2] = calc_worst_total_cost,
    [3] = calc_expected_total_cost,
    [4] = calc_worst_startup_cost,
    [5] = calc_expected_startup_cost,
    [6] = calc_robust_coverage,
    [7] = calc_random_score,
    [8] = calc_postgres_original_score,
    [9] = calc_jointype_based_score,
    [10] = calc_worst_penalty_with_std,
    [11] = calc_expected_penalty_with_std,
    [12] = calc_4th_worst_penalty,
};
