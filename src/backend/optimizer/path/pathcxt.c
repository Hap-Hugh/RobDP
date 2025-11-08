//
// Created by Xuan Chen on 2025/11/5.
//

#include "optimizer/pathcxt.h"

void init_baserel_path_context_1pk(
    PlannerInfo *root,
    RelOptInfo *rel,
    const int round
) {
    Assert(root->pass == 1);
    Assert(round >= 0 && round < error_sample_count);
    Assert(rel->rows_sample != NULL);

    root->round = round;
    /* Do not overwrite. */

    rel->pathlist = rel->pathlist_mat[round];
    rel->partial_pathlist = rel->partial_pathlist_mat[round];
    rel->cheapest_startup_path = rel->cheapest_startup_path_mat[round];
    rel->cheapest_total_path = rel->cheapest_total_path_mat[round];
    rel->cheapest_unique_path = rel->cheapest_unique_path_mat[round];
    rel->cheapest_parameterized_paths = rel->cheapest_parameterized_paths_mat[round];
}

void finalize_baserel_path_context_1p(
    PlannerInfo *root,
    RelOptInfo *rel,
    const int round
) {
    Assert(root->pass == 1);
    Assert(round >= 0 && round < error_sample_count);

    root->round = round;

    rel->pathlist_mat[round] = rel->pathlist;
    rel->partial_pathlist_mat[round] = rel->partial_pathlist;
    rel->cheapest_startup_path_mat[round] = rel->cheapest_startup_path;
    rel->cheapest_total_path_mat[round] = rel->cheapest_total_path;
    rel->cheapest_unique_path_mat[round] = rel->cheapest_unique_path;
    rel->cheapest_parameterized_paths_mat[round] = rel->cheapest_parameterized_paths;
}

void init_baserel_path_context_2p(
    PlannerInfo *root,
    RelOptInfo *rel
) {
    Assert(root->pass == 2);

    root->round = -1;
    /* No need to write back. */

    rel->pathlist = NIL;
    rel->partial_pathlist = NIL;
    rel->cheapest_startup_path = NULL;
    rel->cheapest_total_path = NULL;
    rel->cheapest_unique_path = NULL;
    rel->cheapest_parameterized_paths = NIL;
}

void finalize_baserel_path_context_2p(
    PlannerInfo *root,
    RelOptInfo *rel
) {
    Assert(root->pass == 2);

    root->round = -1;

    rel->pathlist_saved = rel->pathlist;
    rel->partial_pathlist_saved = rel->partial_pathlist;
    rel->cheapest_startup_path_saved = rel->cheapest_startup_path;
    rel->cheapest_total_path_saved = rel->cheapest_total_path;
    rel->cheapest_unique_path_saved = rel->cheapest_unique_path;
    rel->cheapest_parameterized_paths_saved = rel->cheapest_parameterized_paths;
}

void init_joinrel_path_context_1p(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outerrel,
    RelOptInfo *innerrel,
    const int round
) {
    Assert(root->pass == 1);
    Assert(round >= 0 && round < error_sample_count);

    root->round = round;
    /* Overwrite: we only need rows at a particular sample point. */
    joinrel->rows = joinrel->rows_sample->sample_count == 1
                        ? joinrel->saved_rows
                        : joinrel->rows_sample->sample[round];
    if (outerrel) {
        outerrel->rows = outerrel->rows_sample->sample_count == 1
                             ? outerrel->saved_rows
                             : outerrel->rows_sample->sample[round];
    }
    if (innerrel) {
        innerrel->rows = innerrel->rows_sample->sample_count == 1
                             ? innerrel->saved_rows
                             : innerrel->rows_sample->sample[round];
    }

    joinrel->pathlist = joinrel->pathlist_mat[round];
    joinrel->partial_pathlist = joinrel->partial_pathlist_mat[round];
    joinrel->cheapest_startup_path = joinrel->cheapest_startup_path_mat[round];
    joinrel->cheapest_total_path = joinrel->cheapest_total_path_mat[round];
    joinrel->cheapest_unique_path = joinrel->cheapest_unique_path_mat[round];
    joinrel->cheapest_parameterized_paths = joinrel->cheapest_parameterized_paths_mat[round];

    if (outerrel) {
        outerrel->pathlist = outerrel->pathlist_mat[round];
        outerrel->partial_pathlist = outerrel->partial_pathlist_mat[round];
        outerrel->cheapest_startup_path = outerrel->cheapest_startup_path_mat[round];
        outerrel->cheapest_total_path = outerrel->cheapest_total_path_mat[round];
        outerrel->cheapest_unique_path = outerrel->cheapest_unique_path_mat[round];
        outerrel->cheapest_parameterized_paths = outerrel->cheapest_parameterized_paths_mat[round];
    }

    if (innerrel) {
        innerrel->pathlist = innerrel->pathlist_mat[round];
        innerrel->partial_pathlist = innerrel->partial_pathlist_mat[round];
        innerrel->cheapest_startup_path = innerrel->cheapest_startup_path_mat[round];
        innerrel->cheapest_total_path = innerrel->cheapest_total_path_mat[round];
        innerrel->cheapest_unique_path = innerrel->cheapest_unique_path_mat[round];
        innerrel->cheapest_parameterized_paths = innerrel->cheapest_parameterized_paths_mat[round];
    }
}

void finalize_joinrel_path_context_1p(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outerrel,
    RelOptInfo *innerrel,
    const int round
) {
    Assert(root->pass == 1);
    Assert(round >= 0 && round < error_sample_count);

    root->round = round;

    joinrel->pathlist_mat[round] = joinrel->pathlist;
    joinrel->partial_pathlist_mat[round] = joinrel->partial_pathlist;
    joinrel->cheapest_startup_path_mat[round] = joinrel->cheapest_startup_path;
    joinrel->cheapest_total_path_mat[round] = joinrel->cheapest_total_path;
    joinrel->cheapest_unique_path_mat[round] = joinrel->cheapest_unique_path;
    joinrel->cheapest_parameterized_paths_mat[round] = joinrel->cheapest_parameterized_paths;

    if (outerrel) {
        outerrel->pathlist_mat[round] = outerrel->pathlist;
        outerrel->partial_pathlist_mat[round] = outerrel->partial_pathlist;
        outerrel->cheapest_startup_path_mat[round] = outerrel->cheapest_startup_path;
        outerrel->cheapest_total_path_mat[round] = outerrel->cheapest_total_path;
        outerrel->cheapest_unique_path_mat[round] = outerrel->cheapest_unique_path;
        outerrel->cheapest_parameterized_paths_mat[round] = outerrel->cheapest_parameterized_paths;
    }

    if (innerrel) {
        innerrel->pathlist_mat[round] = innerrel->pathlist;
        innerrel->partial_pathlist_mat[round] = innerrel->partial_pathlist;
        innerrel->cheapest_startup_path_mat[round] = innerrel->cheapest_startup_path;
        innerrel->cheapest_total_path_mat[round] = innerrel->cheapest_total_path;
        innerrel->cheapest_unique_path_mat[round] = innerrel->cheapest_unique_path;
        innerrel->cheapest_parameterized_paths_mat[round] = innerrel->cheapest_parameterized_paths;
    }

    joinrel->pathlist = NIL;
    joinrel->partial_pathlist = NIL;
    joinrel->cheapest_startup_path = NULL;
    joinrel->cheapest_total_path = NULL;
    joinrel->cheapest_unique_path = NULL;
    joinrel->cheapest_parameterized_paths = NIL;

    if (outerrel) {
        outerrel->pathlist = NIL;
        outerrel->partial_pathlist = NIL;
        outerrel->cheapest_startup_path = NULL;
        outerrel->cheapest_total_path = NULL;
        outerrel->cheapest_unique_path = NULL;
        outerrel->cheapest_parameterized_paths = NIL;
    }

    if (innerrel) {
        innerrel->pathlist = NIL;
        innerrel->partial_pathlist = NIL;
        innerrel->cheapest_startup_path = NULL;
        innerrel->cheapest_total_path = NULL;
        innerrel->cheapest_unique_path = NULL;
        innerrel->cheapest_parameterized_paths = NIL;
    }
}

void init_joinrel_path_context_2p(
    PlannerInfo *root,
    RelOptInfo *joinrel,
    RelOptInfo *outerrel,
    RelOptInfo *innerrel
) {
    Assert(root->pass == 2);

    root->round = -1;
    /* Write back: we now need expected rows instead of a particular sample point. */
    joinrel->rows = joinrel->saved_rows;
    outerrel->rows = outerrel->saved_rows;
    innerrel->rows = innerrel->saved_rows;

    /* Do not initialize the join relation's paths -- already initialized. */

    /* Initialize the outer relation's paths only when needed. */
    if (outerrel->pathlist == NIL) {
        outerrel->pathlist = outerrel->pathlist_saved;
        outerrel->partial_pathlist = outerrel->partial_pathlist_saved;
        outerrel->cheapest_startup_path = outerrel->cheapest_startup_path_saved;
        outerrel->cheapest_total_path = outerrel->cheapest_total_path_saved;
        outerrel->cheapest_unique_path = outerrel->cheapest_unique_path_saved;
        outerrel->cheapest_parameterized_paths = outerrel->cheapest_parameterized_paths_saved;
    }

    /* Initialize the inner relation's paths only when needed. */
    if (innerrel->pathlist == NIL) {
        innerrel->pathlist = innerrel->pathlist_saved;
        innerrel->partial_pathlist = innerrel->partial_pathlist_saved;
        innerrel->cheapest_startup_path = innerrel->cheapest_startup_path_saved;
        innerrel->cheapest_total_path = innerrel->cheapest_total_path_saved;
        innerrel->cheapest_unique_path = innerrel->cheapest_unique_path_saved;
        innerrel->cheapest_parameterized_paths = innerrel->cheapest_parameterized_paths_saved;
    }
}
