//
// Created by Xuan Chen on 2025/9/2.
//

#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "nodes/pg_list.h"

typedef struct Distribution Distribution;

struct Distribution {
    int sample_count;
    List *probs;
    List *vals;
};

void init_distribution(Distribution *dist);

#endif // DISTRIBUTION_H
