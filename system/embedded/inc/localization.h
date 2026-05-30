#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include "fixed_point.h"
#include <stdint.h>

// === Phase 1: EKF State ===

#define EKF_STATE_SIZE 4

typedef struct {
    fp_t x;
    fp_t y;
    fp_t yaw;
    fp_t v;
} EKFState;

typedef struct {
    fp_t data[EKF_STATE_SIZE * EKF_STATE_SIZE];
} EKFCovariance;

typedef struct {
    fp_t data[EKF_STATE_SIZE * 2];
} EKFObsMatrix;

// === Phase 2: EKF Noise Parameters ===

typedef struct {
    fp_t q_x;
    fp_t q_y;
    fp_t q_yaw;
    fp_t q_v;
    fp_t r_x;
    fp_t r_y;
    fp_t accel_scale;
} EKFParams;

// === Phase 3: EKF Functions ===

void ekf_predict_fp(EKFState* state, EKFCovariance* P,
                    fp_t v_cmd, fp_t yaw_rate_cmd,
                    const EKFParams* params, fp_t dt);

void ekf_update_fp(EKFState* state, EKFCovariance* P,
                   fp_t z_x, fp_t z_y,
                   const EKFParams* params);

// === Phase 4: Default Parameters ===

static const EKFParams EKF_DEFAULTS = {
    0, 0, 0, 0,
    0, 0, 0
};

#endif
