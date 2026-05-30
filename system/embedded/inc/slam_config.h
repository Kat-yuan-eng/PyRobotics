#ifndef SLAM_CONFIG_H
#define SLAM_CONFIG_H

#include "fixed_point.h"
#include "localization.h"
#include <stdint.h>

// === Phase 1: SLAM Constants ===

#define SLAM_MAX_LANDMARKS      20
#define SLAM_MAX_PARTICLES      30
#define SLAM_LM_SIZE            2
#define SLAM_ICP_MAX_ITER       50
#define SLAM_ICP_EPS            FP_FROM_FLOAT(1e-4f)
#define SLAM_MAX_RANGE_FP       FP_FROM_FLOAT(15.0f)
#define SLAM_OBS_SIGMA_R_FP     FP_FROM_FLOAT(0.2f)
#define SLAM_OBS_SIGMA_B_FP     FP_FROM_FLOAT(0.0873f)
#define SLAM_MOTION_SIGMA_V_FP  FP_FROM_FLOAT(0.5f)
#define SLAM_MOTION_SIGMA_Y_FP  FP_FROM_FLOAT(0.1745f)
#define SLAM_N_THRESHOLD_DIV    2

// === Phase 2: Landmark ===

typedef struct {
    fp_t x;
    fp_t y;
    uint8_t observed;
} LandmarkFP;

// === Phase 3: Particle ===

typedef struct {
    fp_t x;
    fp_t y;
    fp_t yaw;
    fp_t w;
    LandmarkFP lm[SLAM_MAX_LANDMARKS];
    fp_t lmP[SLAM_MAX_LANDMARKS * SLAM_LM_SIZE * SLAM_LM_SIZE];
} ParticleFP;

// === Phase 4: SLAM State ===

typedef struct {
    ParticleFP particles[SLAM_MAX_PARTICLES];
    uint16_t n_particles;
    uint16_t n_landmarks;
    fp_t traj_x;
    fp_t traj_y;
    fp_t traj_yaw;
} SLAMState;

// === Phase 5: ICP Result ===

typedef struct {
    fp_t R[4];
    fp_t T[2];
    fp_t final_error;
    uint16_t iterations;
} ICPResultFP;

// === Phase 6: SLAM Functions ===

void slam_init(SLAMState* state, fp_t x0, fp_t y0, fp_t yaw0,
               uint16_t n_particles, uint16_t n_landmarks);

void slam_predict(SLAMState* state, fp_t v_cmd, fp_t yaw_rate_cmd, fp_t dt);

void slam_update(SLAMState* state, const fp_t* observations,
                 uint16_t n_obs, fp_t dt);

void slam_estimate(const SLAMState* state, EKFState* est);

void icp_match_fp(const fp_t* prev_pts, const fp_t* curr_pts,
                  uint16_t n_pts, ICPResultFP* result);

#endif
