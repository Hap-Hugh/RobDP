//
// Created by Xuan Chen on 2025/9/2.
//

#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

struct Distribution;

typedef struct Distribution Distribution;

struct Distribution {
    int sample_count;
    double *probs;
    double *vals;
};

/* Utility functions for making a fake distribution */
Distribution *make_fake_dist(void);

#endif // DISTRIBUTION_H
