//
// Created by Xuan Chen on 2025/10/14.
//

#ifndef PATHUTIL_H
#define PATHUTIL_H

#include "nodes/pathnodes.h"

int count_joinrel_path(
    const PlannerInfo *root,
    const RelOptInfo *rel,
    StringInfo info
);

int write_joinrel_path(
    StringInfo info,
    char* filename
);


#endif // PATHUTIL_H
