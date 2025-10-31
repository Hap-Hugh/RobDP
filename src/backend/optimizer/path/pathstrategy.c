//
// Created by Xuan Chen on 2025/10/31.
//

#define PENALTY_TOLERANCE_FACTOR 1.2

#include "optimizer/pathstrategy.h"

double calc_worst_penalty(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    const int effective
) {
    double worst_penalty = 0.0; /* start from 0 because score is non-negative */
    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_global[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        const double cur_penalty = (cur_sample > cur_thresh) ? (cur_sample - min_global[i]) : 0.0;
        if (cur_penalty > worst_penalty) {
            worst_penalty = cur_penalty;
        }
    }
    return worst_penalty;
}

double calc_expected_penalty(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    const int effective
) {
    double expected_penalty = 0.0; /* start from 0 because score is non-negative */
    for (int i = 0; i < effective; ++i) {
        const double cur_thresh = min_global[i] * PENALTY_TOLERANCE_FACTOR;
        const double cur_sample = total_cost_sample->sample[i];
        const double cur_penalty = (cur_sample > cur_thresh) ? (cur_sample - min_global[i]) : 0.0;
        expected_penalty += cur_penalty;
    }
    expected_penalty /= (double) effective;
    return expected_penalty;
}
