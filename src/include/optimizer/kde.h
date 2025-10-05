//
// Created by Xuan Chen on 2025/9/22.
// Created by Xuan Chen on 2025/9/28.
// Modified by Xuan Chen on 2025/10/2.
// Modified by Xuan Chen on 2025/10/5.
//

#ifndef KDE_H
#define KDE_H

#include "optimizer/sample.h"

/* GUC Parameters */
extern double error_sample_kde_bandwidth;

/* Forward Declarations */
typedef Sample Sample;

typedef ErrorProfile ErrorProfile;

/* ------------------------------- KDE Sampling ------------------------------- */
Sample *make_sample_by_condition(
    const ErrorProfile *ep,
    double sel_est
);

#endif // KDE_H
