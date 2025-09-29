//
// Created by Xuan Chen on 2025/9/28.
//

/*
 * join_rows_distribution_full_enum.c
 *
 * Enumerate ALL combinations (n1 * n2 * n3) from three discrete distributions,
 * then compress to at most target_samples (and <= DIST_MAX_SAMPLE) by taking
 * the top-K by probability and renormalizing.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "postgres.h"
#include "optimizer/dist.h"

/* ------------ Helpers ------------ */

static inline int clamp_int(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Normalize probs to sum to 1. Returns the sum before normalization. */
static double normalize_probs(Distribution *d) {
    double s = 0.0;
    for (int i = 0; i < d->sample_count; ++i) s += d->probs[i];
    if (s > 0.0) {
        double inv = 1.0 / s;
        for (int i = 0; i < d->sample_count; ++i) d->probs[i] *= inv;
    }
    return s;
}

/* A temporary holder for enumerated combinations */
typedef struct {
    double prob; /* product probability */
    double val; /* product value (customize if needed) */
} Combo;

/* qsort comparator: descending by prob */
static int cmp_combo_desc_prob(const void *a, const void *b) {
    const Combo *pa = (const Combo *) a;
    const Combo *pb = (const Combo *) b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

/* Safe product of three ints as size_t, with simple overflow guard. */
static bool safe_mul3_int_to_size_t(int a, int b, int c, size_t *out) {
    if (a <= 0 || b <= 0 || c <= 0) {
        *out = 0;
        return true;
    }
    // promote to size_t and check overflow stepwise
    size_t x = (size_t) a;
    size_t y = (size_t) b;
    size_t z = (size_t) c;
    if (x > SIZE_MAX / y) return false;
    size_t xy = x * y;
    if (xy > SIZE_MAX / z) return false;
    *out = xy * z;
    return true;
}

/* ------------ Core function: full enumeration + compress ------------ */

/*
 * Convenience wrapper:
 *   1) Multiply three input distributions (outer, inner, selectivity) by
 *      enumerating ALL combinations (no pruning).
 *   2) Compress the resulting combinations to 'target_samples' points by
 *      selecting the top-K by probability (K <= DIST_MAX_SAMPLE), and
 *      renormalize probabilities to sum to 1 over the selected K points.
 *
 * Notes:
 * - We never store > DIST_MAX_SAMPLE entries inside any Distribution object.
 * - All N = n1*n2*n3 combinations are held in a separate heap array `Combo*`.
 * - If you prefer NOT to renormalize (keep raw products), see the comment
 *   near the final normalize call and adjust accordingly.
 */
Distribution *
join_rows_distribution(
    const Distribution *outer_rows_dist,
    const Distribution *inner_rows_dist,
    const Distribution *sel_dist,
    int target_samples
) {
    if (!outer_rows_dist || !inner_rows_dist || !sel_dist) return NULL;

    const int n1 = outer_rows_dist->sample_count;
    const int n2 = inner_rows_dist->sample_count;
    const int n3 = sel_dist->sample_count;

    if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
        Distribution *empty = (Distribution *) malloc(sizeof(Distribution));
        if (!empty) return NULL;
        empty->sample_count = 0;
        return empty;
    }

    /* Compute total combinations; allocate a flat array of Combo. */
    size_t N = 0;
    if (!safe_mul3_int_to_size_t(n1, n2, n3, &N)) {
        /* Overflow or too large; you may choose to return NULL or fallback. */
        return NULL;
    }

    Combo *combos = (Combo *) malloc(sizeof(Combo) * N);
    if (!combos) return NULL;

    /* 1) Enumerate ALL combinations. */
    size_t t = 0;
    for (int i = 0; i < n1; ++i) {
        double p1 = outer_rows_dist->probs[i];
        double v1 = outer_rows_dist->vals[i];
        for (int j = 0; j < n2; ++j) {
            double p2 = inner_rows_dist->probs[j];
            double v2 = inner_rows_dist->vals[j];
            const double p12 = p1 * p2;
            const double v12 = v1 * v2; /* customize value rule if needed */
            for (int k = 0; k < n3; ++k) {
                double p3 = sel_dist->probs[k];
                double v3 = sel_dist->vals[k];
                combos[t].prob = p12 * p3; /* product probability */
                combos[t].val = v12 * v3; /* product value      */
                ++t;
            }
        }
    }
    /* Sanity: t == N */
    /* (Optional) If desired, you could compute total_mass over all N here. */

    /* 2) Sort all combinations by probability (descending). */
    qsort(combos, N, sizeof(Combo), cmp_combo_desc_prob);

    /* 3) Determine K and write top-K into output Distribution. */
    int K = clamp_int(target_samples, 1, DIST_MAX_SAMPLE);
    if ((size_t) K > N) K = (int) N;

    Distribution *out = (Distribution *) malloc(sizeof(Distribution));
    if (!out) {
        free(combos);
        return NULL;
    }

    out->sample_count = K;

    /* Copy top-K (note: we will renormalize below). */
    double mass_topk = 0.0;
    for (int i = 0; i < K; ++i) {
        out->probs[i] = combos[i].prob;
        out->vals[i] = combos[i].val;
        mass_topk += combos[i].prob;
    }

    /* 4) Renormalize over the selected K to form a valid compressed distribution.
     *    If you prefer to KEEP raw (un-normalized) product probabilities,
     *    comment out the normalize call and leave as-is.
     */
    (void) mass_topk; /* if you want to use/report it later */
    normalize_probs(out);

    free(combos);
    return out;
}
