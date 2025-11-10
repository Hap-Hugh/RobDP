//
// Created by Xuan Chen on 2025/10/31.
// Modified by Xuan Chen on 2025/11/9.
//

#ifndef PATHSTRATEGY_H
#define PATHSTRATEGY_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"

/* Per-candidate ranking info (only score is relevant here). */
typedef struct PathRank {
    Path *path;
    double score; /* vs. per-type per-sample minima */
} PathRank;

typedef void (*select_path_strategy)(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_worst_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_expected_penalty(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_worst_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_expected_total_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_worst_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_expected_startup_cost(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

extern void calc_robust_cover(
    const List *cand_list,
    PathRank *rank_arr,
    const double *min_envelope,
    int sample_count
);

/*
 * Global array of scoring strategies.
 * Indexed [0..6].
 *
 * Caller can use: select_path_strategy_funcs[i](...)
 */
extern select_path_strategy select_path_strategy_funcs[7];

#endif // PATHSTRATEGY_H
