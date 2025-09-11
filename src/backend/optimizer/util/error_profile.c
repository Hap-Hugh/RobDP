//
// Created by Xuan Chen on 2025/9/11.
//

#include "optimizer/error_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "c.h"

static double clamp01(double x) {
    if (x < 0) return 0;
    if (x > 1) return 1;
    return x;
}

static double maxd(double a, double b) {
    return a > b ? a : b;
}

static double mind(double a, double b) {
    return a < b ? a : b;
}

// 简单均值/方差（无偏方差）
static void mean_std(const double *x, int n, double *mean, double *std) {
    if (n == 0) {
        *mean = 0;
        *std = 0;
        return;
    }
    double m = 0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= (double) n;
    double v = 0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - m;
        v += d * d;
    }
    v = (n > 1) ? v / (double) (n - 1) : 0.0;
    *mean = m;
    *std = sqrt(v);
}

// Silverman 带宽（连续变量 KDE 常用）：h = 1.06 * sigma * n^{-1/5}
static double silverman_bandwidth(double std, int n) {
    if (n < 2 || std <= 0) return 1e-3;
    return 1.06 * std * pow((double) n, -0.2);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *) a, y = *(const double *) b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

// 估计 IQR（用于在 std 很小/数据很尖时控制带宽下限）
static double estimate_iqr(double *tmp, int n) {
    if (n == 0) return 0.0;
    qsort(tmp, n, sizeof(double), cmp_double);
    int q1i = (int) (0.25 * (n - 1));
    int q3i = (int) (0.75 * (n - 1));
    return tmp[q3i] - tmp[q1i];
}

// 高斯核 K(u) = exp(-0.5*u^2) / sqrt(2*pi)；但权重只需要相对值，可省常数
static inline double gaussian_kernel_u(double u) {
    return exp(-0.5 * u * u);
}

// Box-Muller 正态噪声
static double randn(unsigned int *state) {
    // 简易可重入：使用 rand_r
    double u1 = (rand_r(state) + 1.0) / ((double) RAND_MAX + 2.0);
    double u2 = (rand_r(state) + 1.0) / ((double) RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int load_error_profile(const char *base_dir, const char *alias, ErrorProfile *out) {
    if (!base_dir || !alias || !out) return 1;
    memset(out, 0, sizeof(*out));

    // 拼路径 /opt/17a/<alias>.txt
    char path[4096];
    MemSet(path, 0, sizeof(path));
    snprintf(path, sizeof(path), "%s/%s.txt", base_dir, alias);

    FILE *fp = fopen(path, "r");
    if (!fp) return 2;

    // 先粗略计数
    int cap = 1 << 16; // 预分配 65536
    EPSample *buf = (EPSample *) malloc(cap * sizeof(EPSample));
    if (!buf) {
        fclose(fp);
        return 3;
    }

    int n = 0;
    while (1) {
        double t, e;
        int r = fscanf(fp, "%lf %lf", &t, &e);
        if (r == EOF) break;
        if (r != 2) {
            // 跳过坏行
            // 尝试吞掉整行
            int c;
            while ((c = fgetc(fp)) != EOF && c != '\n') {
            }
            continue;
        }
        if (n == cap) {
            cap <<= 1;
            EPSample *nbuf = (EPSample *) realloc(buf, cap * sizeof(EPSample));
            if (!nbuf) {
                free(buf);
                fclose(fp);
                return 4;
            }
            buf = nbuf;
        }
        buf[n].true_sel = clamp01(t);
        buf[n].est_sel = clamp01(e);
        n++;
    }
    fclose(fp);

    if (n == 0) {
        free(buf);
        return 5;
    }

    out->data = buf;
    out->n = n;

    // 统计 std
    double *ests = (double *) malloc(n * sizeof(double));
    double *trus = (double *) malloc(n * sizeof(double));
    if (!ests || !trus) {
        free(ests);
        free(trus);
        free(buf);
        memset(out, 0, sizeof(*out));
        return 6;
    }
    for (int i = 0; i < n; i++) {
        ests[i] = buf[i].est_sel;
        trus[i] = buf[i].true_sel;
    }
    double m, s1, s2;
    mean_std(ests, n, &m, &s1);
    mean_std(trus, n, &m, &s2);
    out->est_std = s1;
    out->true_std = s2;
    free(ests);
    free(trus);
    return 0;
}

void free_error_profile(ErrorProfile *ep) {
    if (!ep) return;
    free(ep->data);
    ep->data = NULL;
    ep->n = 0;
    ep->est_std = 0;
    ep->true_std = 0;
}

typedef struct {
    double d;
    int i;
} Pair;

// 部分排序：简单起见直接全排序
int cmp(const void *a, const void *b) {
    double x = ((const Pair *) a)->d, y = ((const Pair *) b)->d;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static int knn_indices_by_est(const ErrorProfile *ep, double e0, int k, int *out_idx) {
    // 线性选择简单实现（n 可能很大就 O(n log k) 或 O(n log n)），这里用简单排序对的方式
    int n = ep->n;

    Pair *arr = (Pair *) malloc(n * sizeof(Pair));
    if (!arr) return 0;
    for (int i = 0; i < n; i++) {
        arr[i].d = fabs(ep->data[i].est_sel - e0);
        arr[i].i = i;
    }

    qsort(arr, n, sizeof(Pair), cmp);
    int m = (k < n) ? k : n;
    for (int j = 0; j < m; j++) out_idx[j] = arr[j].i;
    free(arr);
    return m;
}

Distribution *build_conditional_distribution(
    const ErrorProfile *ep,
    double e0,
    int n_samples,
    double h_est,
    double h_true,
    unsigned int seed
) {
    if (!ep || ep->n == 0 || n_samples <= 0) return NULL;

    if (seed == 0) seed = (unsigned int) time(NULL);

    // 自动带宽（估计轴）
    if (h_est <= 0) {
        // Silverman + IQR 下限
        double *tmp = (double *) malloc(ep->n * sizeof(double));
        if (!tmp) return NULL;
        for (int i = 0; i < ep->n; i++) tmp[i] = ep->data[i].est_sel;
        double mean, std;
        mean_std(tmp, ep->n, &mean, &std);
        double iqr = estimate_iqr(tmp, ep->n);
        free(tmp);
        double robust_sigma = (iqr > 0) ? mind(std, iqr / 1.349) : std;
        h_est = silverman_bandwidth(robust_sigma, ep->n);
        h_est = maxd(h_est, 1e-3);
    }

    // true 轴的抖动带宽：给个较小比例，避免过度平滑
    if (h_true <= 0) {
        double s = (ep->true_std > 0) ? ep->true_std : 0.05;
        h_true = 0.25 * silverman_bandwidth(s, ep->n);
        h_true = maxd(h_true, 1e-3);
    }

    // 计算权重
    int n = ep->n;
    double *w = (double *) malloc(n * sizeof(double));
    if (!w) return NULL;
    double sumw = 0.0, inv_h = 1.0 / h_est;
    for (int i = 0; i < n; i++) {
        double u = (ep->data[i].est_sel - e0) * inv_h;
        double wi = gaussian_kernel_u(u);
        w[i] = wi;
        sumw += wi;
    }

    // 如果核权重几乎全为 0，回退到 KNN（估计轴上取最近的 k 个）
    int used_knn = 0;
    if (sumw < 1e-12) {
        used_knn = 1;
        int k = (int) fmin((double) n, fmax(50.0, sqrt((double) n))); // 默认取 max(50, sqrt(n))
        int *idx = (int *) malloc(k * sizeof(int));
        if (!idx) {
            free(w);
            return NULL;
        }
        int m = knn_indices_by_est(ep, e0, k, idx);
        // 等权重
        for (int i = 0; i < n; i++) w[i] = 0.0;
        for (int j = 0; j < m; j++) w[idx[j]] = 1.0;
        sumw = (double) m;
        free(idx);
    }

    // 构建累积概率（用于指数查找／二分采样）
    double *cdf = (double *) malloc(n * sizeof(double));
    if (!cdf) {
        free(w);
        return NULL;
    }
    double acc = 0.0;
    if (sumw > 0) {
        for (int i = 0; i < n; i++) {
            acc += w[i] / sumw;
            cdf[i] = acc;
        }
        // 数值误差确保最后为 1
        cdf[n - 1] = 1.0;
    } else {
        // 极端回退：均匀
        for (int i = 0; i < n; i++) {
            cdf[i] = (i + 1) / (double) n;
        }
    }

    // 分配 Distribution
    Distribution *dist = (Distribution *) malloc(sizeof(Distribution));
    if (!dist) {
        free(cdf);
        free(w);
        return NULL;
    }
    dist->sample_count = n_samples;
    dist->probs = (double *) malloc(n_samples * sizeof(double));
    dist->vals = (double *) malloc(n_samples * sizeof(double));
    if (!dist->probs || !dist->vals) {
        free(dist->probs);
        free(dist->vals);
        free(dist);
        free(cdf);
        free(w);
        return NULL;
    }

    // 采样 n_samples 次：先采索引，再在 true 轴加小抖动
    for (int j = 0; j < n_samples; j++) {
        // uniform(0,1)
        double u = (rand_r(&seed) + 1.0) / ((double) RAND_MAX + 2.0);
        // 二分在 cdf 上找第一个 >= u 的位置
        int lo = 0, hi = n - 1, mid;
        while (lo < hi) {
            mid = lo + (hi - lo) / 2;
            if (cdf[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        int pick = lo;
        double t = ep->data[pick].true_sel;

        // 连续化：加高斯抖动（注意截断）
        double t_jitter = t + h_true * randn(&seed);
        t_jitter = clamp01(t_jitter);

        dist->vals[j] = t_jitter;
        dist->probs[j] = 1.0 / (double) n_samples;
    }

    free(cdf);
    free(w);
    return dist;
}

void free_distribution(Distribution *dist) {
    if (!dist) return;
    free(dist->probs);
    free(dist->vals);
    free(dist);
}
