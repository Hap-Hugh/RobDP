//
// Created by Xuan Chen on 2025/10/31.
//

#ifndef PATHSTRATEGY_H
#define PATHSTRATEGY_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"

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

#endif // PATHSTRATEGY_H
