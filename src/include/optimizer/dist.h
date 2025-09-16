//
// Created by Xuan Chen on 2025/9/15.
//

#ifndef DIST_H
#define DIST_H

#include "optimizer/dist.h"
#include "nodes/pathnodes.h"

struct Distribution;
struct EPSample;
struct ErrorProfile;

typedef struct Distribution Distribution;
typedef struct EPSample EPSample;
typedef struct ErrorProfile ErrorProfile;

struct Distribution {
    int sample_count;
    double *probs;
    double *vals;
};

/*
 * One sample in the error profile.
 * true_sel ∈ [0,1]  : the actual selectivity observed
 * est_sel  ∈ [0,1]  : the estimated selectivity reported by the planner
 */
struct EPSample {
    double true_sel;
    double est_sel;
};

/*
 * ErrorProfile
 *  - Stores a collection of (true_sel, est_sel) samples
 *  - Optionally caches some basic statistics (stddevs)
 */
struct ErrorProfile {
    EPSample *data;
    int n;
    double est_std;
    double true_std;
};

Distribution *make_single_point_dist(double val);

/*
 * Load an error profile from "<base_dir>/<alias>.txt".
 * Each line in the file must contain two doubles: <true_sel> <est_sel>.
 *
 * On success:
 *   - fills *out with allocated data and statistics
 *   - returns 0
 * On failure:
 *   - leaves *out zeroed
 *   - returns nonzero error code
 */
int load_error_profile(
    const char *base_dir,
    const char *alias,
    ErrorProfile *out
);

void set_baserel_rows_dist(
    PlannerInfo *root,
    RelOptInfo *rel,
    char *error_profile_path,
    double est_sel
);

void set_joinrel_rows_dist(
    PlannerInfo *root,
    RelOptInfo *rel,
    RelOptInfo *outer_rel,
    RelOptInfo *inner_rel,
    SpecialJoinInfo *sjinfo,
    List *restrictlist,
    char *error_profile_path
);

/*
 * Free resources held by an ErrorProfile.
 * Safe to call on a partially filled or empty profile.
 */
void free_error_profile(
    ErrorProfile *ep
);

/*
 * Build a conditional distribution p(T | Ê = e0) as a sampling approximation.
 * Method:
 *   - Kernel-weighted resampling in the estimated axis (est_sel)
 *   - Gaussian jitter in the true axis (true_sel)
 *
 * Parameters:
 *   ep         : error profile (must be non-null and non-empty)
 *   e0         : conditioning value of estimated selectivity
 *   n_samples  : number of output samples (e.g., 20)
 *   h_est      : bandwidth in the estimated axis; <=0 triggers auto-estimation
 *   h_true     : jitter bandwidth in the true axis; <=0 triggers auto-estimation
 *   seed       : RNG seed (0 uses time(NULL))
 *
 * Returns:
 *   - A newly allocated Distribution* (caller must free with free_distribution)
 *   - NULL on failure
 */
Distribution *build_conditional_distribution(
    const ErrorProfile *ep,
    double e0,
    int n_samples,
    double h_est,
    double h_true,
    unsigned int seed
);

/*
 * Scale all values in a Distribution by a constant factor.
 * Example: convert selectivity samples into row-count samples
 * by multiplying with rel->tuples.
 *
 * Returns:
 *   - A newly allocated Distribution with scaled values
 *   - NULL on failure
 */
Distribution *scale_distribution(
    const Distribution *src,
    double factor
);

/*
 * Free resources held by a Distribution.
 * Safe to call on NULL.
 */
void free_distribution(
    Distribution *dist
);

#endif // DIST_H
