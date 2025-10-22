//
// Created by Xuan Chen on 2025/6/24.
// Modified by Xuan Chen on 2025/7/17.
// Adapted by Xuan Chen on 2025/10/22.
//

#ifndef PATHHINT_H
#define PATHHINT_H

#include "catalog/pg_type.h"

#include "optimizer/planner.h"
#include "utils/lsyscache.h"

struct PathHintItem {
    NodeTag node_tag;
    List *rel_names;
    double total_cost;
    double startup_cost;
};

typedef struct PathHintItem PathHintItem;

struct PathHint {
    int levels_needed;
    double total_cost;
    double startup_cost;
    StringInfo path_hint;
    StringInfo leading_hint;
    List *hint_item_list;
};

typedef struct PathHint PathHint;

struct PathNodeInfo {
    int current_level;
    int levels_needed;
    NodeTag pathtype;
    StringInfo node_name;
    List *rel_names;
    struct PathNodeInfo *parent_info;
    struct PathNodeInfo *inner_info;
    struct PathNodeInfo *outer_info;
    double total_cost;
    double startup_cost;
};

typedef struct PathNodeInfo PathNodeInfo;

void get_path_hint(
    PlannerInfo *planner_info,
    Path *path,
    int levels_needed,
    PathHint *path_hint
);

void traverse_path_hint_tree(
    PlannerInfo *planner_info,
    Path *current_path,
    int current_depth,
    PathNodeInfo **return_info,
    List **hint_list
);

void init_path_node_info(
    PathNodeInfo *node_info,
    const Path *current_path
);

void init_path_hint_item(
    PathHintItem *hint_item,
    const PathNodeInfo *node_info
);

void init_path_hint(
    PathHint *path_hint,
    int levels_needed,
    double total_cost,
    double startup_cost,
    StringInfo path_hint_string,
    StringInfo leading_hint_string,
    List *hint_item_list
);

void finalize_path_node_info(
    const PathNodeInfo *node_info
);

void finalize_path_hint_item(
    PathHintItem *hint_item
);

void finalize_path_hint(
    PathHint *path_hint
);

void freeStringInfo(
    StringInfo string_info
);

Path *get_subpath(
    Path *other_path
);

int compare_rel_name(
    const void *p,
    const void *q
);

int compare_path_hint_item(
    const void *p,
    const void *q
);

void assign_path_node_name(
    const Path *current_path,
    PathNodeInfo *current_info
);

void assign_scan_rel_aliases(
    const PlannerInfo *root_info,
    PathNodeInfo *current_info,
    const Path *current_path
);

void assign_join_rel_aliases(
    PathNodeInfo *node_info,
    const PathNodeInfo *outer_info,
    const PathNodeInfo *inner_info
);

void assign_other_rel_aliases(
    const PlannerInfo *root_info,
    PathNodeInfo *current_info,
    const Path *subpath
);

void build_leading_hint_string(
    const PathNodeInfo *node_info,
    StringInfo leading_hint_string
);

bool path_hint_equals(
    const PathHint *p,
    const PathHint *q
);

void log_path_hint(
    const PathHint *path_hint
);

int get_distinct_path_hint(
    List *path_hint_list,
    StringInfo all_path_hint
);

void get_optimization_info(
    const PathHint *best_path_hint,
    StringInfo all_optimization_info
);

#endif // PATHHINT_H
