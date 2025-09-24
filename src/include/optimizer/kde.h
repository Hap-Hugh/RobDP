//
// Created by Xuan Chen on 2025/9/22.
//

#ifndef KDE_H
#define KDE_H

#include "optimizer/dist.h"

/* GUC Parameters */
extern double error_sample_kde_bandwidth;

/* Forward Declarations */
typedef Distribution Distribution;

typedef ErrorProfile ErrorProfile;

Distribution *build_conditional_distribution(
    const ErrorProfile *ep,
    double est_sel,
    int n_samples,
    double h_est,
    double h_true,
    unsigned int seed
);

#endif // KDE_H
