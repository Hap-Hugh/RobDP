//
// Created by Xuan Chen on 2025/11/18.
//

#ifndef RETENTION_EXT_H
#define RETENTION_EXT_H

#include "nodes/pg_list.h"

extern void
prune_pathlist_by_bucket(
    List **pathlist_ptr,
    bool is_partial,
    int keep_total,
    int keep_startup
);

#endif // RETENTION_EXT_H
