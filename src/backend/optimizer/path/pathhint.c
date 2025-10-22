//
// Created by Xuan Chen on 2025/6/24.
// Modified by Xuan Chen on 2025/7/17.
// Adapted by Xuan Chen on 2025/10/22.
//

#include "postgres.h"
#include "optimizer/pathhint.h"

#include "parser/parsetree.h"

void get_path_hint(
    PlannerInfo *planner_info, Path *path, const int levels_needed, PathHint *path_hint
) {
    // 1. traverse the current path
    List *hint_list = NIL;
    PathNodeInfo *result_info = NULL;

    /* the current depth is different from the current level */
    traverse_path_hint_tree(
        planner_info, path, 0, &result_info, &hint_list
    );

    /* in case we generate nothing in the path list */
    if (!result_info)
        return;

    // 2. make scan and join hint
    List *hint_item_list = NIL;
    const int hint_count = list_length(hint_list);

    // 3. initialize the path hint string (later used by PG Hint Plan)
    StringInfo path_hint_string = makeStringInfo();
    appendStringInfo(path_hint_string, "/*+\n");

    // 4. traverse the path node info items to obtain all needed path hint items
    for (int hint_idx = 0; hint_idx < hint_count; ++hint_idx) {
        const PathNodeInfo *node_info = (PathNodeInfo *) list_nth(hint_list, hint_idx);

        /* skip invalid path node items */
        if (!node_info->node_name || !node_info->rel_names)
            continue;
        /* Do NOT skip other paths here. */

        /* check how many relations we have at current path node */
        const int rel_name_count = list_length(node_info->rel_names);
        /* we want to get a standardized path hint item, so we need to sort the aliases */
        qsort(node_info->rel_names->elements, rel_name_count, sizeof(PathNodeInfo *), compare_rel_name);

        /* generate each path hint item from each path node node_info */
        PathHintItem *hint_item = palloc0(sizeof(PathHintItem));

        /* initialize the path hint item */
        init_path_hint_item(hint_item, node_info);

        /* we need to deep copy all the relation aliases: from each path node node_info to each path hint item */
        for (int rel_idx = 0; rel_idx < rel_name_count; ++rel_idx) {
            const char *alias = (const char *) list_nth(node_info->rel_names, rel_idx);
            hint_item->rel_names = lappend(hint_item->rel_names, pstrdup(alias));
        }

        /* append the path hint item to the path hint item list as the result */
        hint_item_list = lappend(hint_item_list, hint_item);

        /* skip other paths */
        // FIXME: we may have a better method to check here
        if (strcmp(node_info->node_name->data, "OtherPath") == 0)
            continue;

        /* append relation aliases to current path hint item */
        appendStringInfo(path_hint_string, "%s ( ", node_info->node_name->data);
        for (int rel_idx = 0; rel_idx < rel_name_count; ++rel_idx) {
            appendStringInfo(
                path_hint_string, "%s ",
                (const char *) list_nth(node_info->rel_names, rel_idx)
            );
        }
        appendStringInfo(path_hint_string, ")\n");
    }

    /* sort the path hint item list for deduplication (standardization) purposes */
    qsort(hint_item_list->elements, list_length(hint_item_list), sizeof(PathHintItem *), compare_path_hint_item);

    // 5. make leading hint
    StringInfo leading_hint_string = makeStringInfo();
    build_leading_hint_string(result_info, leading_hint_string);

    /* add the leading hint to the path hint string */
    appendStringInfo(path_hint_string, "Leading ( %s )\n", leading_hint_string->data);
    appendStringInfo(path_hint_string, "*/");

    // 6. make path identifier
    init_path_hint(
        path_hint, levels_needed, path->total_cost, path->startup_cost,
        path_hint_string, leading_hint_string, hint_item_list
    );

    // 7. finalize path node info
    ListCell *lc;
    foreach(lc, hint_list) {
        PathNodeInfo *info = (PathNodeInfo *) lfirst(lc);
        finalize_path_node_info(info);
        pfree(info);
    }
    /*
     * Do NOT free `path_hint`, `leading_hint_string`, or `hint_item_list` here.
     * These are stored into the output PathHint* and are owned by the caller.
     *
     * Only clean up temporary PathNodeInfo and hint_list constructed internally.
     */
    pfree(hint_list);
}

void traverse_path_hint_tree(
    PlannerInfo *planner_info,
    Path *current_path,
    const int current_depth,
    PathNodeInfo **return_info,
    List **hint_list
) {
    /* allocate the memory for the current path node info */
    PathNodeInfo *current_info = palloc0(sizeof(PathNodeInfo));
    *hint_list = lappend(*hint_list, current_info);

    /* point the pointer back, since we need it later */
    *return_info = current_info;

    /* initialize the path node info */
    init_path_node_info(current_info, current_path);
    current_info->current_level = current_depth;
    assign_path_node_name(current_path, current_info);

    /* we want to check the type of the current path node (join, scan or other) */
    const NodeTag pathtype = current_path->pathtype;
    if (pathtype == T_NestLoop || pathtype == T_HashJoin || pathtype == T_MergeJoin) {
        /* this is a join path node */
        PathNodeInfo *inner_info = NULL, *outer_info = NULL;
        const JoinPath *current_join_path = (JoinPath *) current_path;

        /* we begin to traverse the outer join path */
        if (current_join_path->outerjoinpath) {
            traverse_path_hint_tree(
                planner_info, current_join_path->outerjoinpath,
                current_depth + 1, &outer_info, hint_list
            );
            /* we should point the pointers back */
            current_info->outer_info = outer_info;
            outer_info->parent_info = current_info;
        }

        /* now let's traverse the inner join path */
        if (current_join_path->innerjoinpath) {
            traverse_path_hint_tree(
                planner_info, current_join_path->innerjoinpath,
                current_depth + 1, &inner_info, hint_list
            );
            /* we should point the pointers back */
            current_info->inner_info = inner_info;
            inner_info->parent_info = current_info;
        }

        /* we want to set the aliases in path node info based on its children nodes */
        assign_join_rel_aliases(current_info, outer_info, inner_info);
    } else if (pathtype == T_SeqScan || pathtype == T_TidScan
               || pathtype == T_IndexScan || pathtype == T_IndexOnlyScan
               || pathtype == T_BitmapIndexScan || pathtype == T_BitmapHeapScan) {
        /* this is a scan path node */
        /* set the aliases in path node info based on itself */
        assign_scan_rel_aliases(planner_info, current_info, current_path);
    } else {
        /* we want to get the subpath for other path */
        // FIXME: we exclude some path types, i.e. AppendPath and MergeAppendPath
        Path *subpath = get_subpath(current_path);

        /* we have and only have one subpath */
        if (subpath != NULL) {
            /* this is a gather path or gather merge path node */
            PathNodeInfo *sub_info = NULL;

            /* now let's traverse its subpath */
            traverse_path_hint_tree(
                planner_info, subpath,
                current_depth + 1, &sub_info, hint_list
            );
            /* the name could be confusing, since the outer info is actually the path node info of its subpath */
            current_info->outer_info = sub_info;
            sub_info->parent_info = current_info;

            /* set the aliases in path node info based on the subpath */
            assign_other_rel_aliases(planner_info, current_info, subpath);
        } else {
            /* at this point, all the needed types are considered; other types are ignored (including its children) */
            elog(LOG, "[OTHER PATH] %d", current_path->pathtype);
        }
    }
}

void init_path_node_info(PathNodeInfo *node_info, const Path *current_path) {
    node_info->rel_names = NIL;
    node_info->node_name = makeStringInfo();
    node_info->parent_info = NULL;
    node_info->inner_info = NULL;
    node_info->outer_info = NULL;
    node_info->total_cost = current_path->total_cost;
    node_info->startup_cost = current_path->startup_cost;
}

void init_path_hint_item(PathHintItem *hint_item, const PathNodeInfo *node_info) {
    hint_item->node_tag = node_info->pathtype;
    hint_item->rel_names = NIL;
    hint_item->total_cost = node_info->total_cost;
    hint_item->startup_cost = node_info->startup_cost;
}

void init_path_hint(
    PathHint *path_hint, const int levels_needed, const double total_cost, const double startup_cost,
    StringInfo path_hint_string, StringInfo leading_hint_string, List *hint_item_list
) {
    path_hint->levels_needed = levels_needed;
    path_hint->total_cost = total_cost;
    path_hint->startup_cost = startup_cost;

    path_hint->path_hint = makeStringInfo();
    appendStringInfo(path_hint->path_hint, "%s", path_hint_string->data);
    path_hint->leading_hint = makeStringInfo();
    appendStringInfo(path_hint->leading_hint, "%s", leading_hint_string->data);
    path_hint->hint_item_list = hint_item_list;
}

void finalize_path_node_info(const PathNodeInfo *node_info) {
    if (!node_info)
        return;
    if (node_info->rel_names) {
        // free each alias string
        ListCell *lc;
        foreach(lc, node_info->rel_names) {
            char *alias = (char *) lfirst(lc);
            pfree(alias);
        }
        list_free(node_info->rel_names);
    }
    freeStringInfo(node_info->node_name);
}

void finalize_path_hint_item(PathHintItem *hint_item) {
    if (!hint_item) {
        return;
    }
    if (hint_item->rel_names) {
        ListCell *lc;
        foreach(lc, hint_item->rel_names) {
            char *alias = (char *) lfirst(lc);
            pfree(alias);
        }
        list_free(hint_item->rel_names);
    }
    pfree(hint_item);
}

void finalize_path_hint(PathHint *path_hint) {
    if (!path_hint)
        return;

    if (path_hint->path_hint)
        freeStringInfo(path_hint->path_hint);
    if (path_hint->leading_hint)
        freeStringInfo(path_hint->leading_hint);

    if (path_hint->hint_item_list) {
        ListCell *lc;
        foreach(lc, path_hint->hint_item_list) {
            PathHintItem *item = (PathHintItem *) lfirst(lc);
            finalize_path_hint_item(item);
        }
        list_free(path_hint->hint_item_list);
    }
}

void freeStringInfo(StringInfo string_info) {
    if (string_info) {
        if (string_info->data) {
            pfree(string_info->data);
        }
        pfree(string_info);
    }
}

Path *get_subpath(Path *other_path) {
    switch (other_path->type) {
        case T_SubqueryScanPath:
            return ((SubqueryScanPath *) other_path)->subpath;
        case T_MaterialPath:
            return ((MaterialPath *) other_path)->subpath;
        case T_MemoizePath:
            return ((MemoizePath *) other_path)->subpath;
        case T_UniquePath:
            return ((UniquePath *) other_path)->subpath;
        case T_GatherPath:
            return ((GatherPath *) other_path)->subpath;
        case T_GatherMergePath:
            return ((GatherMergePath *) other_path)->subpath;
        case T_ProjectionPath:
            return ((ProjectionPath *) other_path)->subpath;
        case T_ProjectSetPath:
            return ((ProjectSetPath *) other_path)->subpath;
        case T_SortPath:
            return ((SortPath *) other_path)->subpath;
        case T_GroupPath:
            return ((GroupPath *) other_path)->subpath;
        case T_UpperUniquePath:
            return ((UpperUniquePath *) other_path)->subpath;
        case T_AggPath:
            return ((AggPath *) other_path)->subpath;
        case T_GroupingSetsPath:
            return ((GroupingSetsPath *) other_path)->subpath;
        case T_WindowAggPath:
            return ((WindowAggPath *) other_path)->subpath;
        case T_SetOpPath:
            return ((SetOpPath *) other_path)->subpath;
        case T_LockRowsPath:
            return ((LockRowsPath *) other_path)->subpath;
        case T_ModifyTablePath:
            return ((ModifyTablePath *) other_path)->subpath;
        case T_LimitPath:
            return ((LimitPath *) other_path)->subpath;
        default:
            elog(LOG, "Unhandled path type %d", other_path->type);
            return NULL;
    }
}

int compare_rel_name(const void *p, const void *q) {
    const char *p_item = *(const char **) p;
    const char *q_item = *(const char **) q;
    return strcmp(p_item, q_item);
}

int compare_path_hint_item(const void *p, const void *q) {
    const PathHintItem *p_item = *(PathHintItem **) p;
    const PathHintItem *q_item = *(PathHintItem **) q;

    const List *p_rel_list = p_item->rel_names;
    const List *q_rel_list = q_item->rel_names;

    const int p_rel_list_length = list_length(p_rel_list);
    const int q_rel_list_length = list_length(q_rel_list);
    if (p_rel_list_length > q_rel_list_length)
        return 1;
    if (p_rel_list_length < q_rel_list_length)
        return -1;

    if (p_item->node_tag > q_item->node_tag)
        return 1;
    if (p_item->node_tag < q_item->node_tag)
        return -1;

    for (int rel_idx = 0; rel_idx < p_rel_list_length; ++rel_idx) {
        const char *p_rel = (const char *) list_nth(p_rel_list, rel_idx);
        const char *q_rel = (const char *) list_nth(q_rel_list, rel_idx);

        const int cmp = strcmp(p_rel, q_rel);
        if (cmp > 0)
            return 1;
        if (cmp < 0)
            return -1;
    }
    return 0;
}

void assign_path_node_name(const Path *current_path, PathNodeInfo *current_info) {
    const char *node_name;
    switch (current_path->pathtype) {
        case T_SeqScan:
            node_name = "SeqScan";
            break;
        case T_TidScan:
            node_name = "TidScan";
            break;
        case T_IndexScan:
            node_name = "IndexScan";
            break;
        case T_IndexOnlyScan:
            node_name = "IndexOnlyScan";
            break;
        case T_BitmapHeapScan:
        case T_BitmapIndexScan:
            node_name = "BitmapScan";
            break;
        case T_NestLoop:
            node_name = "NestLoop";
            break;
        case T_HashJoin:
            node_name = "HashJoin";
            break;
        case T_MergeJoin:
            node_name = "MergeJoin";
            break;
        default:
            node_name = "OtherPath";
            break;
    }
    current_info->pathtype = current_path->pathtype;
    appendStringInfo(current_info->node_name, "%s", node_name);
}

void assign_scan_rel_aliases(
    const PlannerInfo *root_info,
    PathNodeInfo *current_info,
    const Path *current_path
) {
    const Index relid = current_path->parent->relid;
    Assert(relid != 0);

    const RangeTblEntry *rte = planner_rt_fetch(relid, root_info);
    const char *aliasname = NULL;

    if (rte->alias && rte->alias->aliasname)
        aliasname = rte->alias->aliasname;
    else if (rte->eref && rte->eref->aliasname)
        aliasname = rte->eref->aliasname;

    char *alias = pstrdup(aliasname ? aliasname : "<unknown>");
    current_info->rel_names = lappend(current_info->rel_names, alias);
}

void assign_join_rel_aliases(
    PathNodeInfo *node_info,
    const PathNodeInfo *outer_info,
    const PathNodeInfo *inner_info
) {
    if (outer_info != NULL) {
        /* if the left path exists */
        const int rel_name_count = list_length(outer_info->rel_names);
        for (int rel_name_idx = 0; rel_name_idx < rel_name_count; ++rel_name_idx) {
            const char *alias = (const char *) list_nth(outer_info->rel_names, rel_name_idx);
            node_info->rel_names = lappend(node_info->rel_names, pstrdup(alias));
        }
    }
    if (inner_info != NULL) {
        /* if the right path exists */
        const int rel_name_count = list_length(inner_info->rel_names);
        for (int rel_name_idx = 0; rel_name_idx < rel_name_count; ++rel_name_idx) {
            const char *alias = (const char *) list_nth(inner_info->rel_names, rel_name_idx);
            node_info->rel_names = lappend(node_info->rel_names, pstrdup(alias));
        }
    }
}

void assign_other_rel_aliases(
    const PlannerInfo *root_info,
    PathNodeInfo *current_info,
    const Path *subpath
) {
    int relid = 0;
    const Bitmapset *relids = subpath->parent->relids;

    while ((relid = bms_next_member(relids, relid)) >= 0) {
        Assert(relid > 0);

        const RangeTblEntry *rte = planner_rt_fetch(relid, root_info);
        const char *aliasname = NULL;

        if (rte->alias && rte->alias->aliasname)
            aliasname = rte->alias->aliasname;
        else if (rte->eref && rte->eref->aliasname)
            aliasname = rte->eref->aliasname;

        char *alias = pstrdup(aliasname ? aliasname : "<unknown>");
        current_info->rel_names = lappend(current_info->rel_names, alias);
    }
}

void build_leading_hint_string(const PathNodeInfo *node_info, StringInfo leading_hint_string) {
    if (node_info == NULL)
        return;
    switch (node_info->pathtype) {
        case T_SeqScan:
        case T_TidScan:
        case T_IndexScan:
        case T_IndexOnlyScan:
        case T_BitmapIndexScan:
            // scan node must be a leaf
            appendStringInfo(leading_hint_string, "%s", (const char *) linitial(node_info->rel_names));
            break;
        case T_NestLoop:
        case T_HashJoin:
        case T_MergeJoin:
            // append its left and right child
            appendStringInfo(leading_hint_string, "( ");
            build_leading_hint_string(node_info->outer_info, leading_hint_string);
            appendStringInfo(leading_hint_string, " ");
            build_leading_hint_string(node_info->inner_info, leading_hint_string);
            appendStringInfo(leading_hint_string, " )");
            break;
        default:
            // go to its child directly
            build_leading_hint_string(node_info->outer_info, leading_hint_string);
            break;
    }
}

bool path_hint_equals(const PathHint *p, const PathHint *q) {
    if (p == NULL || q == NULL)
        return false;
    if (p->levels_needed != q->levels_needed)
        return false;

    const List *p_hint = p->hint_item_list;
    const List *q_hint = q->hint_item_list;

    const int p_hint_length = list_length(p_hint);
    const int q_hint_length = list_length(q_hint);
    if (p_hint_length != q_hint_length)
        return false;
    if (strcmp(p->leading_hint->data, q->leading_hint->data) != 0)
        return false;

    for (int hint_item_idx = 0; hint_item_idx < p_hint_length; ++hint_item_idx) {
        const PathHintItem *p_item = (PathHintItem *) list_nth(p_hint, hint_item_idx);
        const PathHintItem *q_item = (PathHintItem *) list_nth(q_hint, hint_item_idx);

        if (p_item->node_tag != q_item->node_tag)
            return false;

        const List *p_rel_list = p_item->rel_names;
        const List *q_rel_list = q_item->rel_names;

        const int p_rel_list_length = list_length(p_rel_list);
        const int q_rel_list_length = list_length(q_rel_list);
        if (p_rel_list_length != q_rel_list_length)
            return false;

        for (int rel_idx = 0; rel_idx < p_rel_list_length; ++rel_idx) {
            const char *p_rel = (const char *) list_nth(p_rel_list, rel_idx);
            const char *q_rel = (const char *) list_nth(q_rel_list, rel_idx);

            if (strcmp(p_rel, q_rel) != 0)
                return false;
        }
    }
    return true;
}

void log_path_hint(const PathHint *path_hint) {
    elog(LOG, "[PATH HINT] %d, Leading( %s ), (%.3f..%.3f)",
         path_hint->levels_needed, path_hint->leading_hint->data,
         path_hint->startup_cost, path_hint->total_cost);

    const List *hint_item_string_list = path_hint->hint_item_list;
    const int hint_item_count = list_length(hint_item_string_list);
    for (int hint_item_idx = 0; hint_item_idx < hint_item_count; ++hint_item_idx) {
        const PathHintItem *item = (PathHintItem *) list_nth(hint_item_string_list, hint_item_idx);

        StringInfo item_string = makeStringInfo();
        appendStringInfo(item_string, "\t[%d] %d ", hint_item_idx, item->node_tag);

        const List *rel_list = item->rel_names;
        const int rel_name_count = list_length(item->rel_names);
        for (int rel_name_idx = 0; rel_name_idx < rel_name_count; ++rel_name_idx)
            appendStringInfo(item_string, "%s ", (const char *) list_nth(rel_list, rel_name_idx));

        appendStringInfo(item_string, "(%.3f..%.3f)", item->startup_cost, item->total_cost);
        elog(LOG, "%s", item_string->data);
        freeStringInfo(item_string);
    }
}

int get_distinct_path_hint(List *path_hint_list, StringInfo all_path_hint) {
    if (path_hint_list == NIL)
        return -1;

    const int path_hint_count = list_length(path_hint_list);
    elog(LOG, "[PATH HINT] count = %d", path_hint_count);

    List *seen_path_hints = NIL;

    ListCell *lc_path_hint;
    foreach(lc_path_hint, path_hint_list) {
        PathHint *path_hint = (PathHint *) lfirst(lc_path_hint);

        if (path_hint->path_hint != NULL) {
            bool already_seen = false;

            ListCell *lc_seen_hint;
            foreach(lc_seen_hint, seen_path_hints) {
                PathHint *seen = (PathHint *) lfirst(lc_seen_hint);
                if (strcmp(seen->path_hint->data, path_hint->path_hint->data) == 0) {
                    already_seen = true;
                    break;
                }
            }

            if (!already_seen) {
                seen_path_hints = lappend(seen_path_hints, path_hint);
                appendStringInfo(
                    all_path_hint, "-- (%.3f..%.3f)\n", path_hint->startup_cost, path_hint->total_cost
                );
                appendStringInfo(
                    all_path_hint, "%s\n\n", path_hint->path_hint->data
                );
            }
        }
    }
    const int distinct_path_hint_count = list_length(seen_path_hints);
    if (seen_path_hints)
        list_free(seen_path_hints);
    return distinct_path_hint_count;
}

void get_optimization_info(const PathHint *best_path_hint, StringInfo all_optimization_info) {
    if (best_path_hint == NULL)
        return;

    List *hint_item_list = best_path_hint->hint_item_list;
    if (hint_item_list == NIL)
        return;

    /* We need to know: how many relations are there in the final relation? */
    int final_rel_count = 0;
    const int hint_item_count = list_length(hint_item_list);
    /* The array index starts at 0, so we need hint item `count + 1`. */
    double *total_cost_array = palloc0(sizeof(double) * (hint_item_count + 1));
    /*
     * Note: not all elements in this array will store a valid total cost.
     * First, we may encounter a bushy plan. Second, the number of hint items is
     * often greater than the number of base relations, since we create scan
     * paths for each relation, but we only care about tracking the maximum total
     * cost per relation.
     */

    /* We need to traverse the hint item list and get total cost of each relation. */
    ListCell *lc;
    foreach(lc, hint_item_list) {
        const PathHintItem *hint_item = (PathHintItem *) lfirst(lc);
        const int rel_count = list_length(hint_item->rel_names);
        final_rel_count = Max(final_rel_count, rel_count);
        const double total_cost = hint_item->total_cost;
        total_cost_array[rel_count] = Max(total_cost, total_cost_array[rel_count]);
    }

    /* We would like to keep the maximum total cost of each level in the optimal path. */
    double max_total_cost = 0.0;
    for (int rel_count = 1; rel_count <= hint_item_count && rel_count <= final_rel_count; ++rel_count) {
        const double total_cost = total_cost_array[rel_count];
        max_total_cost = Max(max_total_cost, total_cost);
        appendStringInfo(all_optimization_info, "%d, %.3f\n", rel_count, total_cost);
    }
    appendStringInfo(all_optimization_info, "%d, %.3f\n", -1, max_total_cost);

    pfree(total_cost_array);
}
