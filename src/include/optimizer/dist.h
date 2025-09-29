//
// Created by Xuan Chen on 2025/9/22.
// Modified by Xuan Chen on 2025/9/24.
//

#ifndef DIST_H
#define DIST_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "nodes/pg_list.h"

#define EP_MAX_SAMPLE     1024
#define DIST_MAX_SAMPLE   128
#define EP_MAX_BIN        8

/* GUC Parameters */
extern bool enable_rows_dist;
extern int error_sample_count;
extern int error_sample_seed;

/* Forward Declarations */
typedef struct Distribution Distribution;

typedef struct ErrorProfileSample ErrorProfileSample;

typedef struct ErrorDistParams ErrorDistParams;

typedef struct ErrorProfile ErrorProfile;

struct Distribution {
    int sample_count;
    double probs[DIST_MAX_SAMPLE];
    double vals[DIST_MAX_SAMPLE];
};

struct ErrorProfileSample {
    double sel_true;
    double sel_est;
};

/* Parameters of each bin’s error distribution */
struct ErrorDistParams {
    int bin_index;
    int n_points; /* number of raw samples in this bin */
    double bandwidth_h; /* KDE bandwidth (if needed later) */
    double mean_logratio; /* mean of log(sel_true/sel_est) */
    double std_logratio; /* std of log(sel_true/sel_est) */
    double sel_est_lo; /* left boundary of sel_est */
    double sel_est_hi; /* right boundary of sel_est */
};

struct ErrorProfile {
    int sample_count;
    double std_true;
    double std_est;
    double error_dist_thresh[EP_MAX_BIN]; /* sel_est threshold of each bin */
    ErrorDistParams params[EP_MAX_BIN]; /* parameters of each bin */
    Distribution error_dist[EP_MAX_BIN]; /* discrete distributions (optional) */
    ErrorProfileSample data[EP_MAX_SAMPLE]; /* input samples */
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

Distribution *make_single_point_dist(
    double val
);

Distribution *scale_distribution(
    const Distribution *src,
    double factor
);

Distribution *get_conditional_distribution(
    const ErrorProfile *ep,
    double sel_est
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
