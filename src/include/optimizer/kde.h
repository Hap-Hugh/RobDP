//
// Created by Xuan Chen on 2025/9/22.
// Created by Xuan Chen on 2025/9/28.
//

#ifndef KDE_H
#define KDE_H

#include "optimizer/dist.h"

/* GUC Parameters */
extern double error_sample_kde_bandwidth;
extern int error_bin_count;

/* Forward Declarations */
typedef Distribution Distribution;

typedef ErrorProfile ErrorProfile;

void calc_error_dist(
    ErrorProfile *ep
);

int find_bin_by_sel_est(
    const ErrorProfile *ep,
    double sel_est
);

#endif // KDE_H
