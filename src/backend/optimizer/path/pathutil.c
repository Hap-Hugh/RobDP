//
// Created by Xuan Chen on 2025/10/14.
//

#include "postgres.h"
#include "nodes/bitmapset.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathutil.h"

#include <unistd.h>
#include <sys/fcntl.h>

#include "lib/stringinfo.h"
#include "parser/parsetree.h"

/* ----------------------------------------------------------------
 * Helper: return a display name for an RTE.
 *
 * We prefer rte->eref->aliasname because it reflects user-visible
 * aliases (e.g., "t1", "subq") across many RTE kinds. For plain
 * tables, eref->aliasname usually equals the relname unless the
 * query wrote an explicit alias.
 * ----------------------------------------------------------------
 */
static const char *rte_display_name(const RangeTblEntry *rte) {
    if (rte && rte->eref && rte->eref->aliasname)
        return rte->eref->aliasname;

    /* Fallback: a generic placeholder if eref is unavailable. */
    return "<rte>";
}

/* ----------------------------------------------------------------
 * Helper: append a human-readable join key like "(a ⋈ b ⋈ c)"
 * from a RelOptInfo's relids. Requires PlannerInfo to resolve
 * RT indexes (1-based) to RangeTblEntry names.
 * ----------------------------------------------------------------
*/
static void append_relids_join_name(
    StringInfo buf,
    const PlannerInfo *root,
    const Bitmapset *relids
) {
    bool first = true;

    appendStringInfoChar(buf, '(');

    /* bms_next_member() is non-destructive; iterate with prevbit state. */
    for (int prev = -1;;) {
        const int rtindex = bms_next_member(relids, prev); /* next 0-based member, or <0 when done */
        if (rtindex < 0)
            break;

        prev = rtindex; /* advance the cursor */

        const RangeTblEntry *rte = planner_rt_fetch(rtindex, root);
        const char *name = rte_display_name(rte);

        if (!first)
            appendStringInfoString(buf, " ⋈ ");
        appendStringInfoString(buf, name);
        first = false;
    }

    appendStringInfoChar(buf, ')');
}

/*
 * count_joinrel_path
 *
 * Count paths on a given RelOptInfo and append a line like:
 *   "(a ⋈ b ⋈ c ⋈ d), <normal_count>, <partial_count>\n"
 *
 * Notes:
 * - For baserel, rel->relid != 0 and rel->relids has exactly one bit set.
 * - For joinrel, rel->relid == 0 and rel->relids has multiple bits set.
 * - We do not rely on rel->relid for naming; we always render from relids.
 */
int count_joinrel_path(const PlannerInfo *root, const RelOptInfo *rel, StringInfo info) {
    int normal_count = 0;
    int partial_count = 0;

    /* Sanity checks */
    if (rel == NULL || info == NULL || root == NULL)
        return -1;

    /* Count normal paths (all standard candidates) */
    if (rel->pathlist != NIL)
        normal_count = list_length(rel->pathlist);

    /* Count partial paths (parallel-aware candidates) */
    if (rel->partial_pathlist != NIL)
        partial_count = list_length(rel->partial_pathlist);

    /* Build the human-readable "(a ⋈ b ⋈ ...)" name from relids */
    StringInfoData namebuf;
    initStringInfo(&namebuf);
    append_relids_join_name(&namebuf, root, rel->relids);

    /* Emit the requested line format */
    appendStringInfo(info, "%s, %d, %d\n", namebuf.data, normal_count, partial_count);

    /* Cleanup */
    pfree(namebuf.data);

    return normal_count + partial_count;
}

/*
 * write_joinrel_path
 *
 * Write StringInfo contents to a file.
 * - If filename is NULL or empty, log to server log instead.
 * - Append mode: if the file exists, new data will be appended.
 */
int write_joinrel_path(StringInfo info, char *filename) {
    /* basic checks */
    if (info == NULL || info->data == NULL)
        return -1;

    /* if filename is not given, just log */
    if (filename == NULL || filename[0] == '\0') {
        elog(LOG, "%s", info->data);
        return 0;
    }

    /* open file (O_APPEND ensures atomic append behavior on most systems) */
    int fd = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd < 0) {
        ereport(WARNING,
                (errcode_for_file_access(),
                    errmsg("could not open \"%s\" for writing: %m", filename)));
        elog(LOG, "%s", info->data);
        return -1;
    }

    /* write content */
    ssize_t written = write(fd, info->data, info->len);
    if (written < 0) {
        ereport(WARNING,
                (errcode_for_file_access(),
                    errmsg("could not write to \"%s\": %m", filename)));
        close(fd);
        return -1;
    }

    /* optionally append a newline if not already present */
    if (info->len == 0 || info->data[info->len - 1] != '\n')
        write(fd, "\n", 1);

    /* close file */
    close(fd);

    return 0;
}
