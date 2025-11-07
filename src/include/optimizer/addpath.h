//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/29.
//

#ifndef ADDPATH_H
#define ADDPATH_H

#include "pathstrategy.h"
#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"
#include "nodes/pg_list.h"

List *
add_path_by_strategy(
    const PlannerInfo *root,
    int lev_index,
    int rel_index,
    path_score_strategy add_path_func,
    int add_path_limit,
    int sample_count,
    bool is_partial
);

List *
retain_path_by_strategy(
    const PlannerInfo *root,
    int lev_index,
    int rel_index,
    List *cand_list,
    path_score_strategy retain_path_func,
    int retain_path_limit,
    int sample_count,
    bool is_partial
);

void
calc_min_score_from_pathlist(
    RelOptInfo *joinrel
);

List **
calc_minimum_envelope(
    List *saved_join_rel_levels,
    int sample_count,
    int levels_needed
);

List *
sort_pathlist_by_total_cost(
    List *pathlist
);

Cost
get_best_path_total_cost(
    const RelOptInfo *final_rel
);

#endif // ADDPATH_H
