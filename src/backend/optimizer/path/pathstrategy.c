//
// Created by Xuan Chen on 2025/10/31.
//

#define PENALTY_TOLERANCE_FACTOR 1.2

#include "optimizer/pathstrategy.h"

/*
 * calc_worst_penalty
 *
 * Compute the worst (maximum) penalty among samples.
 *
 * Penalty is measured relative to the global minimum vector:
 *
 *   penalty(j) = (v(j) > min_global(j) * PENALTY_TOLERANCE_FACTOR)
 *                  ? (v(j) - min_global(j))
 *                  : 0.0
 *
 * This metric reflects the worst-case deviation from the baseline.
 * Zero means the path is comparable to the best observed path.
 */
double
calc_worst_penalty(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_global,
    const int effective
) {
    /* Score >= 0 always */
    double worst_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_global != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_global[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        const double cur_penalty =
                (cur_sample > cur_thresh) ? (cur_sample - min_global[i]) : 0.0;

        if (cur_penalty > worst_penalty)
            worst_penalty = cur_penalty;
    }
    return worst_penalty;
}

/*
 * calc_expected_penalty
 *
 * Compute the average penalty across all samples.
 *
 * This metric reflects expected (mean) deviation from the baseline,
 * instead of worst-case deviation.
 */
double
calc_expected_penalty(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_global,
    const int effective
) {
    double expected_penalty = 0.0;

    Assert(total_cost_sample != NULL);
    Assert(min_global != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_global[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        expected_penalty +=
                (cur_sample > cur_thresh) ? (cur_sample - min_global[i]) : 0.0;
    }
    return expected_penalty / (double) effective;
}

/*
 * calc_worst_total_cost
 *
 * Return the maximum value among total_cost samples.
 *
 * This corresponds to the worst-case total execution cost.
 */
double
calc_worst_total_cost(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_global, /* unused */
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
 * calc_expected_total_cost
 *
 * Return the average total cost across samples.
 *
 * This corresponds to expected execution cost.
 */
double
calc_expected_total_cost(
    const Sample *startup_cost_sample, /* unused */
    const Sample *total_cost_sample,
    const double *min_global, /* unused */
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
 * calc_worst_startup_cost
 *
 * Return the maximum startup cost among samples.
 *
 * This is the worst-case startup cost, useful for pipelining latency mode.
 */
double
calc_worst_startup_cost(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample, /* unused */
    const double *min_global, /* unused */
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
 * calc_expected_startup_cost
 *
 * Return the average startup cost across samples.
 *
 * This reflects expected first-tuple latency for pipelined plans.
 */
double
calc_expected_startup_cost(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample, /* unused */
    const double *min_global, /* unused */
    const int effective
) {
    double expected_startup_cost = 0.0;

    Assert(startup_cost_sample != NULL);
    Assert(effective > 0);

    for (int i = 0; i < effective; ++i)
        expected_startup_cost += startup_cost_sample->sample[i];

    return expected_startup_cost / (double) effective;
}

/* Global strategy array */
path_score_strategy path_score_strategies[6] = {
    calc_worst_penalty,
    calc_expected_penalty,
    calc_worst_total_cost,
    calc_expected_total_cost,
    calc_worst_startup_cost,
    calc_expected_startup_cost
};
