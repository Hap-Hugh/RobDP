//
// Created by Xuan Chen on 2025/10/18.
//

#ifndef ADDPATH_H
#define ADDPATH_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"
#include "nodes/pg_list.h"

void
consider_additional_path(
    List **pathlist_ptr,
    List *additional_pathlist,
    int sample_count,
    int mp_path_limit
);

#endif // ADDPATH_H
