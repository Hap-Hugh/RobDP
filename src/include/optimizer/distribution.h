//
// Created by Xuan Chen on 2025/9/2.
//

#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "nodes/pathnodes.h"

typedef Distribution Distribution;

typedef struct SelErrorDistEntry SelDistributionEntry;

typedef struct GlobalSelErrorDistInfo GlobalSelDistributionInfo;

extern GlobalSelDistributionInfo *global_info;

struct Distribution {
    int sample_count;
    double *probs;
    double *vals;
};

struct SelErrorDistEntry {
    int rel_count;
    char *names;
    char *keys;
    Distribution *sel_error_dist;
};

struct GlobalSelErrorDistInfo {
    List *sel_error_dist_entries;
};

/* Utility functions for getting base relation aliases */
char *get_baserel_alias(
    PlannerInfo *root,
    Index relid
);

/* Utility functions for getting join relation aliases */
char *get_joinrel_aliases(
    PlannerInfo *root,
    Relids relids
);

/* Utility functions for making a fake distribution */
Distribution *make_fake_dist(void);

/*
 * Setting the global error distribution info
 * Note: This function stores error distribution for base relations'
 * and 2-way-join relations' selectivity contained on estimated selectivity.
 */
void set_global_sel_error_dist_info(void);

/*
 * Read error distribution of selectivity from file.
 * Note: This function focuses on mocking such distributions.
 * Those are distributions conditioned on a given estimated selectivity.
 */
Distribution *fake_sel_error_dist_from_file(
    Index relid
);

/*
 * We would like to generate a distribution of real selectivity as output
 * given an estimated selectivity as input.
 */
Distribution *fake_baserel_real_sel_from_sel_error(
    PlannerInfo *root,
    Index relid
);

/*
 * We would like to calculate a distribution of rows as output
 * given the distribution of real selectivity as input.
 */
void calc_baserel_rows_from_real_sel(
    PlannerInfo *root,
    RelOptInfo *baserel
);

/*
 * We would like to calculate the path's rows distribution from the
 * the baser relation's rows distribution or from the parameterized path
 * information.
 */
void set_scan_path_rows_dist_with_ppi(
    RelOptInfo *baserel,
    Path *scan_path,
    ParamPathInfo *param_info
);

#endif // DISTRIBUTION_H
