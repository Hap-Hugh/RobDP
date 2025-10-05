//
// Created by Xuan Chen on 2025/10/2.
// Modified by Xuan Chen on 2025/10/5.
//

#ifndef EP_H
#define EP_H

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/sample.h"

#define EP_MAX_SAMPLE     1024
#define EP_MAX_BIN        8

/* Forward Declarations */
typedef struct Sample Sample;

typedef struct ErrorProfileRaw ErrorProfileRaw;

typedef struct ErrorProfile ErrorProfile;

struct ErrorProfileRaw {
    double sel_true;
    double sel_est;
};

struct ErrorProfile {
    int sample_count;
    double std_true;
    double std_est;
    ErrorProfileRaw data[EP_MAX_SAMPLE]; /* input raw data */
};

/* ------------------------------- Aliases ------------------------------- */
char *get_alias(
    const PlannerInfo *root,
    Index relid
);

char *get_std_alias(
    const PlannerInfo *root,
    Index relid
);

/* ------------------------------- Error Profiles ------------------------------- */
int read_error_profile(
    const char *filename,
    ErrorProfile *ep
);

bool get_error_profile(
    const char *alias,
    const char *alias_fallback,
    ErrorProfile **ep
);

#endif // EP_H
