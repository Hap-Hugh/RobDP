//
// Created by Xuan Chen on 2025/9/2.
//

#include "postgres.h"
#include "optimizer/distribution.h"

Distribution *make_fake_dist(void) {
    elog(LOG, "make_fake_dist::[begin]");

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

    elog(LOG, "make_fake_dist::[end]");
    return dist;
}
