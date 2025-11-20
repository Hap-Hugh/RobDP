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

extern void calc_robust_coverage(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope /* unused */,
    const int sample_count
) {
    elog(ERROR, "`calc_robust_coverage` cannot be used here.");
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
