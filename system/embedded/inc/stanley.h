#ifndef STANLEY_H
#define STANLEY_H

#include "fixed_point.h"
#include <stdint.h>

// === Phase 1: Data Structures ===

#define STANLEY_MAX_PATH 50

typedef struct {
    fp_t x[STANLEY_MAX_PATH];
    fp_t y[STANLEY_MAX_PATH];
    fp_t theta[STANLEY_MAX_PATH];
    fp_t kappa[STANLEY_MAX_PATH];
    uint16_t n;
} PathBuffer;

typedef struct {
    fp_t steer_rad;
    fp_t e_lat;
    fp_t e_heading;
    uint16_t nearest_idx;
    uint16_t preview_idx;
} StanleyResult;

// === Phase 2: Stanley Controller ===

StanleyResult stanley_steer(const PathBuffer* path, fp_t speed, fp_t k_e, fp_t k_v,
                            fp_t wheelbase, fp_t k_preview,
                            fp_t steer_prev, fp_t max_steer_rate_rad_per_s,
                            fp_t dead_zone, fp_t v_damping,
                            fp_t max_steer_rad, fp_t dt);

#endif
