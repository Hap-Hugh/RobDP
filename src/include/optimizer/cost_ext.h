//
// Created by Xuan Chen on 2025/11/1.
//

#ifndef COST_EXT_H
#define COST_EXT_H

#include "nodes/pathnodes.h"

/* ==== ==== ==== ==== ==== ==== SCAN COST ==== ==== ==== ==== ==== ==== */

extern void get_restriction_qual_cost(
    PlannerInfo *root,
    const RelOptInfo *baserel,
    const ParamPathInfo *param_info,
    QualCost *qpqual_cost
);

extern void cost_seqscan_1p(
    Path *path,
    PlannerInfo *root,
    const RelOptInfo *baserel,
    const ParamPathInfo *param_info
);

extern void cost_seqscan_2p(
    Path *path,
    PlannerInfo *root,
    const RelOptInfo *baserel,
    const ParamPathInfo *param_info
);

extern void cost_index_1p(
    IndexPath *path,
    PlannerInfo *root,
    double loop_count,
    bool partial_path
);

extern void cost_index_2p(
    IndexPath *path,
    PlannerInfo *root,
    double loop_count,
    bool partial_path
);

/* ==== ==== ==== ==== ==== ==== JOIN COST ==== ==== ==== ==== ==== ==== */

#define GET_ROW(s, i, scalar, is_const) \
( ((s) == NULL || (s)->sample_count <= 0) \
? (scalar) : ((is_const) ? (s)->sample[0] : (s)->sample[(i)]) )

#define GET_COST(s, i, scalar, is_const) \
( ((s) == NULL || (s)->sample_count <= 0) \
? (scalar) : ((is_const) ? (s)->sample[0] : (s)->sample[(i)]) )


extern double get_parallel_divisor(
    const Path *path
);

extern double approx_tuple_count(
    PlannerInfo *root,
    const JoinPath *path,
    List *quals
);

extern int32 get_expr_width(
    const PlannerInfo *root,
    const Node *expr
);

extern double relation_byte_size(
    double tuples,
    int width
);

extern double page_size(
    double tuples,
    int width
);

extern bool has_indexed_join_quals(
    NestPath *path
);

extern void initial_cost_nestloop_1p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const Path *outer_path,
    Path *inner_path,
    const JoinPathExtraData *extra
);

extern void final_cost_nestloop_1p(
    PlannerInfo *root,
    NestPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
);

extern void initial_cost_mergejoin_1p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    List *mergeclauses,
    Path *outer_path,
    Path *inner_path,
    List *outersortkeys,
    List *innersortkeys,
    JoinPathExtraData *extra
);

extern void final_cost_mergejoin_1p(
    PlannerInfo *root,
    MergePath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
);

extern void initial_cost_hashjoin_1p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const List *hashclauses,
    Path *outer_path,
    Path *inner_path,
    JoinPathExtraData *extra,
    bool parallel_hash
);

extern void final_cost_hashjoin_1p(
    PlannerInfo *root,
    HashPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
);

extern void initial_cost_nestloop_2p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const Path *outer_path,
    Path *inner_path,
    const JoinPathExtraData *extra
);

extern void final_cost_nestloop_2p(
    PlannerInfo *root,
    NestPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
);

extern void initial_cost_mergejoin_2p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    List *mergeclauses,
    Path *outer_path,
    Path *inner_path,
    List *outersortkeys,
    List *innersortkeys,
    JoinPathExtraData *extra
);

extern void final_cost_mergejoin_2p(
    PlannerInfo *root,
    MergePath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
);

extern void initial_cost_hashjoin_2p(
    PlannerInfo *root,
    JoinCostWorkspace *workspace,
    JoinType jointype,
    const List *hashclauses,
    Path *outer_path,
    Path *inner_path,
    JoinPathExtraData *extra,
    bool parallel_hash
);

extern void final_cost_hashjoin_2p(
    PlannerInfo *root,
    HashPath *path,
    JoinCostWorkspace *workspace,
    const JoinPathExtraData *extra
);

#endif // COST_EXT_H
