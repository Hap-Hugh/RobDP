//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
// Modified by Xuan Chen on 2025/10/2.
// Modified by Xuan Chen on 2025/10/3.
//

#ifndef SAMPLE_H
#define SAMPLE_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "nodes/pg_list.h"

#define DIST_MAX_SAMPLE   128

/* GUC Parameters */
extern bool enable_rows_dist;
extern int error_sample_count;
extern int error_sample_seed;

/* Forward Declarations */
typedef struct Sample Sample;

typedef struct ErrorProfileRaw ErrorProfileRaw;

typedef struct ErrorSampleParams ErrorSampleParams;

typedef struct ErrorProfile ErrorProfile;

struct Sample {
    int sample_count;
    double sample[DIST_MAX_SAMPLE];
};

/* ------------------------------- Utilities ------------------------------- */
double clamp01(
    double sel
);

void calc_mean_std(
    const double *array,
    int n_samples,
    double *res_mean,
    double *res_std
);

/* ------------------------------- Samples ------------------------------- */
Sample *make_sample_by_single_value(
    double val
);

Sample *make_sample_by_scale_factor(
    const Sample *src,
    double factor
);

Sample *make_sample_by_bin(
    const ErrorProfile *ep,
    double sel_est
);

Sample *make_single_sample_by_join_sample(
    const Sample *outer_rows_sample,
    const Sample *inner_rows_sample,
    const Sample *sel_sample
);

/* ------------------------------- Relations ------------------------------- */
void set_baserel_rows_sample(
    const PlannerInfo *root,
    RelOptInfo *baserel,
    double sel_est
);

void set_joinrel_rows_sample(
    const PlannerInfo *root,
    RelOptInfo *joinrel,
    const RelOptInfo *outer_rel,
    const RelOptInfo *inner_rel,
    List *restrictlist,
    double sel_est
);

#endif // SAMPLE_H
