//
// Created by Xuan Chen on 2025/9/28.
// Modified by Xuan Chen on 2025/9/30.
//

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "postgres.h"
#include "optimizer/dist.h"

Distribution *
join_rows_distribution(
    const Distribution *outer_rows_dist,
    const Distribution *inner_rows_dist,
    const Distribution *sel_dist,
    int target_samples
) {
    /* Each input must be either a scalar (sample_count==1) or have target_samples */
    Assert(outer_rows_dist->sample_count == 1 || outer_rows_dist->sample_count == target_samples);
    Assert(inner_rows_dist->sample_count == 1 || inner_rows_dist->sample_count == target_samples);
    Assert(sel_dist->sample_count == 1 || sel_dist->sample_count == target_samples);

    /* Whether to broadcast each distribution as a constant */
    const bool outer_to_constant = outer_rows_dist->sample_count == 1;
    const bool inner_to_constant = inner_rows_dist->sample_count == 1;
    const bool sel_to_constant = sel_dist->sample_count == 1;

    /* Constant values (used when sample_count==1) */
    const double outer_const_val = outer_rows_dist->vals[0];
    const double inner_const_val = inner_rows_dist->vals[0];
    const double sel_const_val = sel_dist->vals[0];

    /* Constant probabilities (used when sample_count==1) */
    const double outer_const_prob = outer_rows_dist->probs[0];
    const double inner_const_prob = inner_rows_dist->probs[0];
    const double sel_const_prob = sel_dist->probs[0];

    /* Fast path: if all are constants, multiply once and return a single-point distribution */
    if (outer_to_constant && inner_to_constant && sel_to_constant) {
        Distribution *join_dist = make_single_point_dist(
            outer_const_val * inner_const_val * sel_const_val
        );
        return join_dist;
    }

    /* Allocate result with target_samples */
    Distribution *join_dist = palloc0(sizeof(Distribution));
    join_dist->sample_count = target_samples;

    /* Element-wise multiply across i; broadcast any scalar inputs */
    double acc_prob = 0.0;
    for (int i = 0; i < target_samples; ++i) {
        double current_prob = 1.0;
        double current_val = 1.0;

        /* Outer: use constant when broadcasting, otherwise use i-th sample */
        if (outer_to_constant) {
            current_prob *= outer_const_prob;
            current_val *= outer_const_val;
        } else {
            current_prob *= outer_rows_dist->probs[i];
            current_val *= outer_rows_dist->vals[i];
        }

        /* Inner: use constant when broadcasting, otherwise use i-th sample */
        if (inner_to_constant) {
            current_prob *= inner_const_prob;
            current_val *= inner_const_val;
        } else {
            current_prob *= inner_rows_dist->probs[i];
            current_val *= inner_rows_dist->vals[i];
        }

        /* Selectivity: use constant when broadcasting, otherwise use i-th sample */
        if (sel_to_constant) {
            current_prob *= sel_const_prob;
            current_val *= sel_const_val;
        } else {
            current_prob *= sel_dist->probs[i];
            current_val *= sel_dist->vals[i];
        }

        join_dist->probs[i] = current_prob;
        join_dist->vals[i] = current_val;
        acc_prob += current_prob;
    }

    /*
     * Normalize probabilities:
     * - If sum is positive, scale to make them sum to 1.0
     * - If sum is zero (e.g., all inputs had zero probs), fall back to uniform
     */
    Assert(acc_prob >= 0.0 && acc_prob <= 1.0);
    Distribution *scaled_join_dist = scale_distribution(join_dist, 1.0 / acc_prob);
    pfree(join_dist);

    return scaled_join_dist;
}
