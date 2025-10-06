//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
//

#ifndef DIST_H
#define DIST_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "nodes/pg_list.h"

#define EP_MAX_SAMPLE 1024

/* GUC Parameters */
extern bool enable_rows_dist;
extern int error_sample_count;
extern int error_sample_seed;

/* Forward Declarations */
typedef struct Distribution Distribution;

typedef struct ErrorProfileRaw ErrorProfileRaw;

typedef struct ErrorProfile ErrorProfile;

struct Distribution {
    int sample_count;
    double *probs;
    double *vals;
};

struct ErrorProfileRaw {
    double sel_true;
    double sel_est;
};

struct ErrorProfile {
    int sample_count;
    double std_true;
    double std_est;
    ErrorProfileRaw data[EP_MAX_SAMPLE];
};

double clamp01(
    double sel
);

char *get_alias(
    const PlannerInfo *root,
    Index relid
);

char *get_std_alias(
    const PlannerInfo *root,
    Index relid
);

void calc_mean_std(
    const double *array,
    int n_samples,
    double *res_mean,
    double *res_std
);

Distribution *make_dist_by_single_value(
    double val
);

Distribution *make_dist_by_scale_factor(
    const Distribution *src,
    double factor
);

void free_distribution(
    Distribution *dist
);

int read_error_profile(
    const char *filename,
    ErrorProfile *ep
);

bool get_error_profile(
    const char *alias,
    const char *alias_fallback,
    ErrorProfile **ep
);

void set_baserel_rows_dist(
    const PlannerInfo *root,
    RelOptInfo *rel,
    double sel_est
);

Distribution *join_rows_distribution(
    const Distribution *outer_rows_dist,
    const Distribution *inner_rows_dist,
    const Distribution *sel_dist,
    int target_samples
);

void set_joinrel_rows_dist(
    const PlannerInfo *root,
    RelOptInfo *rel,
    const RelOptInfo *outer_rel,
    const RelOptInfo *inner_rel,
    List *restrictlist,
    double sel_est
);

#endif // DIST_H
