#ifndef FIXED_POINT_H
#define FIXED_POINT_H

#include <stdint.h>
#include <stdlib.h>

// === Phase 1: Q16.16 Type ===

typedef int32_t fp_t;

#define FP_FRAC_BITS  16
#define FP_ONE        (1 << FP_FRAC_BITS)
#define FP_HALF       (FP_ONE >> 1)
#define FP_PI         205887
#define FP_TWO_PI     411775
#define FP_HALF_PI    102944
#define FP_RAD_TO_DEG 3754936
#define FP_DEG_TO_RAD 1144
#define FP_EPSILON    66

// === Phase 2: Conversion ===

#define FP_FROM_INT(x)    ((fp_t)((x) << FP_FRAC_BITS))
#define FP_FROM_FLOAT(x)  ((fp_t)((x) * FP_ONE))
#define FP_TO_INT(x)      ((int32_t)((x) >> FP_FRAC_BITS))
#define FP_TO_FLOAT(x)    ((float)(x) / FP_ONE)

// === Phase 3: Arithmetic ===

#define FP_MUL(a, b)  ((fp_t)(((int64_t)(a) * (b)) >> FP_FRAC_BITS))
#define FP_DIV(a, b)  ((fp_t)(((int64_t)(a) << FP_FRAC_BITS) / (b)))

// === Phase 4: Utility ===

#define FP_ABS(x)     (((x) < 0) ? (-(x)) : (x))
#define FP_MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define FP_MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define FP_CLIP(x, lo, hi) (FP_MAX(lo, FP_MIN(x, hi)))

static inline fp_t fp_sqrt(fp_t x) {
    if (x <= 0) return 0;
    fp_t s = x;
    fp_t s_prev;
    for (int i = 0; i < 16; i++) {
        s_prev = s;
        s = (s + FP_DIV(x, s)) >> 1;
        if (FP_ABS(s - s_prev) <= 1) break;
    }
    return s;
}

#endif
