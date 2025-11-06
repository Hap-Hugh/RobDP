//
// Created by Xuan Chen on 2025/10/31.
//

#ifndef PATHSTRATEGY_H
#define PATHSTRATEGY_H

#include "postgres.h"
#include "nodes/pathnodes.h"

#include "optimizer/sample.h"

/* Forward Declarations */
typedef struct Sample Sample;

typedef double (*path_score_strategy)(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    int effective
);

double calc_worst_penalty(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    int effective
);

double calc_expected_penalty(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    int effective
);

double calc_worst_total_cost(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    int effective
);

double calc_expected_total_cost(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    int effective
);

double calc_worst_startup_cost(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    int effective
);

double calc_expected_startup_cost(
    const Sample *startup_cost_sample,
    const Sample *total_cost_sample,
    const double *min_global,
    int effective
);

/*
 * Global array of scoring strategies.
 * Indexed [0..5].
 *
 * Caller can use: path_score_strategies[i](...)
 */
extern path_score_strategy path_score_strategies[6];

#endif // PATHSTRATEGY_H
