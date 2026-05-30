#include "vehicle_control.h"
#include "lookup_table.h"

// === Phase 1: Curvature Speed Limit ===

static fp_t curvature_speed_limit(fp_t v_nominal, fp_t kappa, fp_t a_lat_max) {
    fp_t kappa_safe = FP_ABS(kappa) + FP_FROM_FLOAT(0.001);
    fp_t v_sq = FP_DIV(a_lat_max, kappa_safe);
    if (v_sq < 0) v_sq = 0;
    fp_t v_curv = fp_sqrt(v_sq);
    return FP_MIN(v_nominal, v_curv);
}

// === Phase 2: Radians To Degrees ===

static fp_t rad_to_deg_fp(fp_t rad) {
    return FP_MUL(rad, FP_RAD_TO_DEG);
}

// === Phase 3: Vehicle Control ===

ControlCommand vehicle_control(const PathBuffer* path, fp_t speed_actual,
                               fp_t v_nominal, fp_t max_steer_deg,
                               fp_t k_e, fp_t k_v, fp_t a_lat_max,
                               fp_t wheelbase, fp_t k_preview,
                               fp_t max_steer_rate_dps,
                               fp_t dead_zone, fp_t v_damping,
                               fp_t jerk_max, fp_t brake_ramp_rate, fp_t d_alpha,
                               const ControlState* prev_state,
                               ControlState* new_state,
                               fp_t dt) {
    ControlCommand cmd;
    cmd.control_valid = 0;
    cmd.emergency = 0;
    cmd.steering_deg = 0;
    cmd.throttle = 0;
    cmd.brake = 0;
    cmd.target_speed = 0;
    cmd.lateral_error = 0;
    cmd.heading_error_deg = 0;

    if (path->n == 0) {
        cmd.emergency = 1;
        return cmd;
    }

    fp_t max_steer_rad = FP_MUL(max_steer_deg, FP_DEG_TO_RAD);
    fp_t max_steer_rate_rad = FP_MUL(FP_FROM_FLOAT(1.745e-2f), max_steer_rate_dps);

    StanleyResult stan = stanley_steer(path, speed_actual, k_e, k_v,
                                       wheelbase, k_preview,
                                       prev_state->steer_prev,
                                       max_steer_rate_rad,
                                       dead_zone, v_damping,
                                       max_steer_rad, dt);

    uint16_t idx = stan.nearest_idx;
    fp_t kappa = path->kappa[idx];
    fp_t v_target = curvature_speed_limit(v_nominal, kappa, a_lat_max);

    fp_t a_actual_est = 0;

    PIDResult pid = pid_cascade(v_target, speed_actual, a_actual_est,
                                FP_FROM_FLOAT(1.0f), FP_FROM_FLOAT(0.05f), FP_FROM_FLOAT(0.1f),
                                FP_FROM_FLOAT(0.8f), FP_FROM_FLOAT(0.1f),
                                FP_FROM_INT(3), FP_FROM_INT(2),
                                d_alpha, jerk_max,
                                brake_ramp_rate, prev_state->brake_prev,
                                &prev_state->outer_state, &prev_state->inner_state,
                                dt);

    new_state->outer_state = pid.outer;
    new_state->inner_state = pid.inner;
    new_state->steer_prev = stan.steer_rad;
    new_state->a_prev = pid.a_target;
    new_state->brake_prev = pid.brake;

    cmd.steering_deg = rad_to_deg_fp(stan.steer_rad);
    cmd.throttle = pid.throttle;
    cmd.brake = pid.brake;
    cmd.target_speed = v_target;
    cmd.lateral_error = stan.e_lat;
    cmd.heading_error_deg = rad_to_deg_fp(stan.e_heading);
    cmd.control_valid = 1;

    return cmd;
}
