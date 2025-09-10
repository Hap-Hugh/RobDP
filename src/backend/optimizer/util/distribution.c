//
// Created by Xuan Chen on 2025/9/2.
//

#include "postgres.h"
#include "optimizer/distribution.h"

Distribution *make_fake_dist(double factor) {
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

    dist->vals[0] = factor * 0.8;
    dist->vals[1] = factor * 0.9;
    dist->vals[2] = factor * 1.1;
    dist->vals[3] = factor * 1.0;
    dist->vals[4] = factor * 0.6;

    elog(LOG, "make_fake_dist::[end]");
    return dist;
}

Distribution *make_single_point_dist(double val) {
    elog(LOG, "make_single_point_dist::[begin]");

    Distribution *dist = palloc0(sizeof(Distribution));
    dist->sample_count = 1;
    dist->probs = palloc0(sizeof(double));
    dist->vals = palloc0(sizeof(double));

    /* Single Point distribution */
    dist->probs[0] = val;
    dist->vals[0] = 1.0;

    elog(LOG, "make_single_point_dist::[end]");
    return dist;
}
