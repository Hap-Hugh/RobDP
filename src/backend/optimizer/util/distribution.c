//
// Created by Xuan Chen on 2025/9/2.
//

#include "postgres.h"
#include "optimizer/distribution.h"
#include "optimizer/session_mem.h"
#include "parser/parsetree.h"

/*
 * Pointer for global selectivity distribution information
 * There is only one instance per session.
 */
GlobalSelDistributionInfo *global_info = NULL;

char *get_baserel_alias(PlannerInfo *root, Index relid) {
    MemoryContext global_session_context = session_mem_context();
    MemoryContext old_context = MemoryContextSwitchTo(global_session_context);

    Assert(relid > 0);

    RangeTblEntry *rte = planner_rt_fetch(relid, root);
    const char *aliasname = NULL;

    /* Prefer explicit alias, fall back to eref->aliasname */
    if (rte->alias && rte->alias->aliasname)
        aliasname = rte->alias->aliasname;
    else if (rte->eref && rte->eref->aliasname)
        aliasname = rte->eref->aliasname;

    char *alias = pstrdup(aliasname ? aliasname : "<unknown>");
    MemoryContextSwitchTo(old_context);
    return alias;
}

char *get_joinrel_aliases(PlannerInfo *root, Relids relids) {
    MemoryContext global_session_context = session_mem_context();
    MemoryContext old_context = MemoryContextSwitchTo(global_session_context);

    StringInfoData buf;
    int member = -1;
    bool first = true;

    initStringInfo(&buf);

    if (relids == NULL || bms_is_empty(relids))
        return pstrdup("");

    while ((member = bms_next_member(relids, member)) >= 0) {
        Index relid = (Index) member + 1;
        char *alias = get_baserel_alias(root, relid);

        if (!first)
            appendStringInfoChar(&buf, ' ');
        appendStringInfoString(&buf, alias ? alias : "<unknown>");

        if (alias)
            pfree(alias);
        first = false;
    }

    char *res = pstrdup(buf.data);
    pfree(buf.data);

    MemoryContextSwitchTo(old_context);
    return res;
}

Distribution *make_fake_dist(void) {
    elog(LOG, "make_fake_dist::[begin]");

    MemoryContext global_session_context = session_mem_context();
    MemoryContext old_context = MemoryContextSwitchTo(global_session_context);

    int sample_count = 5;

    Distribution *dist = palloc0(sizeof(Distribution));
    dist->sample_count = sample_count;
    dist->probs = palloc0(sizeof(double) * sample_count);
    dist->vals = palloc0(sizeof(double) * sample_count);

    /* Fake distribution */
    dist->probs[0] = 0.1;
    dist->probs[1] = 0.2;
    dist->probs[2] = 0.3;
    dist->probs[3] = 0.2;
    dist->probs[4] = 0.2;

    dist->vals[0] = 0.1;
    dist->vals[1] = 0.01;
    dist->vals[2] = 0.001;
    dist->vals[3] = 0.02;
    dist->vals[4] = 0.05;

    MemoryContextSwitchTo(old_context);
    elog(LOG, "make_fake_dist::[end]");
    return dist;
}

void set_global_sel_error_dist_info(void) {
    elog(LOG, "set_global_sel_error_dist_info::[begin]");

    MemoryContext global_session_context = session_mem_context();
    MemoryContext old_context = MemoryContextSwitchTo(global_session_context);

    /* 1. Allocate memory for the global selectivity distribution information */
    global_info = palloc0(sizeof(GlobalSelDistributionInfo));
    global_info->sel_error_dist_entries = NIL;

    /* 2. Make a fake selectivity distribution entry */
    // FIXME: We need to read from real files.
    Distribution *dist = fake_sel_error_dist_from_file(1);
    SelDistributionEntry *entry = palloc0(sizeof(SelDistributionEntry));
    entry->rel_count = 1;
    entry->names = pstrdup("rel");
    entry->keys = pstrdup("key");
    entry->sel_error_dist = dist;

    /* 3. Allocate memory for one fake selectivity distribution entry */
    global_info->sel_error_dist_entries = lappend(global_info->sel_error_dist_entries, entry);

    MemoryContextSwitchTo(old_context);
    elog(LOG, "set_global_sel_error_dist_info::[end]");
}

Distribution *fake_sel_error_dist_from_file(Index relid) {
    elog(LOG, "fake_sel_error_dist_from_file::[begin] relid = %d", relid);

    Distribution *dist = make_fake_dist();

    elog(LOG, "fake_sel_error_dist_from_file::[end] relid = %d", relid);
    return dist;
}

/*
 * Given the conditioned selectivity error distribution and the estimated selectivity,
 * we would like to get the base relations' real selectivity distribution.
 */
Distribution *fake_baserel_real_sel_from_sel_error(PlannerInfo *root, Index relid) {
    elog(LOG, "fake_baserel_real_sel_from_sel_error::[begin] relid = %d", relid);
    Assert(relid > 0);
    MemoryContext global_session_context = session_mem_context();
    MemoryContext old_context = MemoryContextSwitchTo(global_session_context);

    // FIXME: Now we are using a fake distribution for all base relations.
    if (global_info == NULL || global_info->sel_error_dist_entries == NIL) {
        elog(ERROR, "fake_baserel_real_sel_from_sel_error: global_info == NULL");
    }
    char *names = get_baserel_alias(root, relid);
    int count = list_length(global_info->sel_error_dist_entries);
    for (int i = 0; i < count; ++i) {
        SelDistributionEntry *entry = list_nth(global_info->sel_error_dist_entries, i);
        if (strcmp(names, entry->names) == 0) {
            elog(LOG, "fake_baserel_real_sel_from_sel_error::[match] names = %s", names);
            // FIXME: We should take at least `entry->sel_error_dist` as input
            return make_fake_dist();
        }
    }
    elog(LOG, "fake_baserel_real_sel_from_sel_error::[no_match] names = %s", names);

    /* TODO: Now we always use the first entry */
    SelDistributionEntry *entry = lfirst(list_head(global_info->sel_error_dist_entries));
    // FIXME: We should take at least `entry->sel_error_dist` as input
    Distribution *dist = make_fake_dist();

    MemoryContextSwitchTo(old_context);
    elog(LOG, "fake_baserel_real_sel_from_sel_error::[end] relid = %d", relid);
    return dist;
}

/* No sampling. We only do basic calculating here. */
void calc_baserel_rows_from_real_sel(PlannerInfo *root, RelOptInfo *baserel) {
    elog(LOG, "calc_baserel_rows_from_real_sel::[begin] relid = %d", baserel->relid);

    MemoryContext global_session_context = session_mem_context();
    MemoryContext old_context = MemoryContextSwitchTo(global_session_context);

    /* We would like to duplicate the distribution. */
    double tuples = baserel->tuples;
    Distribution *real_sel_dist = fake_baserel_real_sel_from_sel_error(root, baserel->relid);
    Distribution *rows_dist = palloc0(sizeof(Distribution));
    memcpy(rows_dist, real_sel_dist, sizeof(Distribution));

    int samples = rows_dist->sample_count;
    for (int i = 0; i < samples; ++i) {
        rows_dist->vals[i] *= tuples;
    }
    baserel->rows_dist = rows_dist;

    MemoryContextSwitchTo(old_context);
    elog(LOG, "calc_baserel_rows_from_real_sel::[end] relid = %d", baserel->relid);
}

void set_scan_path_rows_dist_with_ppi(
    RelOptInfo *baserel, Path *scan_path, ParamPathInfo *param_info
) {
    elog(LOG, "set_scan_path_rows_dist_with_ppi::[begin] relid = %d", baserel->relid);
    Assert(relid > 0);
    if (baserel->rows_dist == NULL) {
        elog(ERROR, "set_scan_path_rows_dist_with_ppi::[no_baserel_dist] relid = %d", baserel->relid);
    }
    MemoryContext global_session_context = session_mem_context();
    MemoryContext old_context = MemoryContextSwitchTo(global_session_context);

    /* If there is no `param_info`, then the only thing that we need to do is to duplicate
     * the path's rows distribution */
    Distribution *new_rows_dist = palloc0(sizeof(Distribution));
    memcpy(new_rows_dist, baserel->rows_dist, sizeof(Distribution));
    double baserel_rows = baserel->rows;

    /* Now we will use the parameterized path information for path's rows distribution estimation */
    if (param_info) {
        double param_rows = param_info->ppi_rows;
        double factor = param_rows / baserel_rows;
        int count = new_rows_dist->sample_count;
        /* We would like to calibrate the estimation of the path's rows */
        for (int i = 0; i < count; ++i) {
            new_rows_dist->vals[i] *= factor;
        }
    }
    scan_path->rows_dist = new_rows_dist;

    MemoryContextSwitchTo(old_context);
    elog(LOG, "set_scan_path_rows_dist_with_ppi::[end] relid = %d", baserel->relid);
}
