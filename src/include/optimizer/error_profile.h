//
// Created by Xuan Chen on 2025/9/11.
//

#ifndef ERROR_PROFILE_H
#define ERROR_PROFILE_H

#include "distribution.h"

typedef struct EPSample {
    double true_sel; // T \in [0,1]
    double est_sel; // Ê \in [0,1]
} EPSample;

typedef struct ErrorProfile {
    EPSample *data;
    int n;
    // 可缓存一些统计量
    double est_std;
    double true_std;
} ErrorProfile;

// 读取 /opt/17a/<alias>.txt 到 ErrorProfile
// 返回 0 成功；非 0 失败
int load_error_profile(
    const char *base_dir, const char *alias, ErrorProfile *out
);

// 释放 ErrorProfile
void free_error_profile(ErrorProfile *ep);

// 生成条件分布 p(T | Ê = e0) 的抽样近似（核加权自助法）
// n_samples: 生成的样本数（如 20）
// h_est: 估计轴的核带宽（<=0 则自动估计）
// h_true: 对 true 值添加的高斯抖动带宽（<=0 则自动估计一个较小值）
// seed: 随机种子（0 用内部熵）
// 返回 Distribution*（需调用 free_distribution 释放）；失败返回 NULL
Distribution *build_conditional_distribution(
    const ErrorProfile *ep,
    double e0,
    int n_samples,
    double h_est,
    double h_true,
    unsigned int seed
);

// 释放 Distribution
void free_distribution(Distribution *dist);

#endif // ERROR_PROFILE_H
