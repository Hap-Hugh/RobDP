//
// Created by Xuan Chen on 2025/10/31.
// Modified by Xuan Chen on 2025/11/9.
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
    const double *min_envelope, /* unused */
    const int sample_count
) {
    // const int path_count = list_length(cand_list);
    // double cost_array[path_count][sample_count];
    // memset(cost_array, 0, sizeof(cost_array));
    //
    // ListCell *lc;
    // int idx = 0;
    // foreach(lc, cand_list) {
    //     const Path *path = lfirst(lc);
    //     Sample *sample = path.
    //     Assert(path_count > 0);
    //     path->total_cost_sample;
    //     ++idx;
    // }
    //
    // for (int i = 0; i < path_count; ++i) {
    // }
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
    const double *min_envelope, /* unused */
    const int sample_count /* unused */
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

/* Global strategy array */
select_path_strategy select_path_strategy_funcs[10] = {
    [0] = calc_worst_penalty,
    [1] = calc_expected_penalty,
    [2] = calc_worst_total_cost,
    [3] = calc_expected_total_cost,
    [4] = calc_worst_startup_cost,
    [5] = calc_expected_startup_cost,
    [6] = calc_robust_coverage,
    [7] = calc_random_score,
    [8] = calc_postgres_original_score,
    [9] = calc_jointype_based_score
};
