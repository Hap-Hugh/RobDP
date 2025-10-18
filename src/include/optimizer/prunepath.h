//
// Created by Xuan Chen on 2025/10/18.
//

#ifndef PRUNEPATH_H
#define PRUNEPATH_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"
#include "nodes/pg_list.h"

void
prune_path(
    List **pathlist_ptr, int sample_count,
    int mc_path_limit, int mp_path_limit
);

#endif // PRUNEPATH_H
