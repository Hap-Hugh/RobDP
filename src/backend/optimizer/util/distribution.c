//
// Created by Xuan Chen on 2025/9/2.
//

#include "postgres.h"
#include "optimizer/distribution.h"

void init_distribution(Distribution *dist) {
    dist->sample_count = 0;
    dist->probs = NIL;
    dist->vals = NIL;
}
