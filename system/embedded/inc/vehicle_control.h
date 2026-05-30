#ifndef VEHICLE_CONTROL_H
#define VEHICLE_CONTROL_H

#include "fixed_point.h"
#include "stanley.h"
#include "pid.h"
#include "localization.h"
#include "slam_config.h"
#include <stdint.h>

// === Phase 1: Control Output ===

typedef struct {
    fp_t steering_deg;
    fp_t throttle;
    fp_t brake;
    fp_t target_speed;
    fp_t lateral_error;
    fp_t heading_error_deg;
    uint8_t control_valid;
    uint8_t emergency;
} ControlCommand;

typedef struct {
    PIDState outer_state;
    PIDState inner_state;
    fp_t steer_prev;
    fp_t a_prev;
    fp_t brake_prev;
} ControlState;

// === Phase 3: Vehicle State ===

typedef struct {
    EKFState ekf_state;
    EKFCovariance ekf_P;
    EKFParams ekf_params;
    SLAMState slam_state;
    ControlState ctrl_state;
    uint8_t localization_valid;
    uint8_t slam_valid;
} VehicleState;

// === Phase 2: Top-Level Control ===

ControlCommand vehicle_control(const PathBuffer* path, fp_t speed_actual,
                               fp_t v_nominal, fp_t max_steer_deg,
                               fp_t k_e, fp_t k_v, fp_t a_lat_max,
                               fp_t wheelbase, fp_t k_preview,
                               fp_t max_steer_rate_dps,
                               fp_t dead_zone, fp_t v_damping,
                               fp_t jerk_max, fp_t brake_ramp_rate, fp_t d_alpha,
                               const ControlState* prev_state,
                               ControlState* new_state,
                               fp_t dt);

#endif
