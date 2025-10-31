//
// Created by Xuan Chen on 2025/10/18.
// Modified by Xuan Chen on 2025/10/29.
//

#ifndef ADDPATH_H
#define ADDPATH_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"
#include "nodes/pg_list.h"

void
reconsider_pathlist(
    const PlannerInfo *root,
    int lev_index,
    int rel_index,
    int sample_count,
    int mp_path_limit,
    bool is_partial
);

void
calc_score_from_pathlist(
    RelOptInfo *joinrel,
    int sample_count,
    bool is_partial
);

void
calc_final_score_from_pathlist(
    RelOptInfo *joinrel
);

#endif // ADDPATH_H
