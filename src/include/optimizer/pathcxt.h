//
// Created by Xuan Chen on 2025/11/5.
//

#ifndef PATHCXT_H
#define PATHCXT_H

#include "optimizer/sample.h"

extern void init_baserel_path_context_1pk(
    PlannerInfo *root,
    RelOptInfo *rel,
    int round
);

extern void finalize_baserel_path_context_1p(
    PlannerInfo *root,
    RelOptInfo *rel,
    int round
);

extern void init_baserel_path_context_2p(
    PlannerInfo *root,
    RelOptInfo *rel
);

extern void finalize_baserel_path_context_2p(
    PlannerInfo *root,
    RelOptInfo *rel
);

extern void init_joinrel_path_context_1p(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outerrel,
    RelOptInfo *innerrel,
    int round
);

extern void finalize_joinrel_path_context_1p(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outerrel,
    RelOptInfo *innerrel,
    int round
);

extern void init_joinrel_path_context_2p(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outerrel,
    RelOptInfo *innerrel
);

#endif // PATHCXT_H
