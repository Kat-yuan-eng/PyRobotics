#include "pid.h"

// === Phase 1: Single PID Step ===

static fp_t pid_step(fp_t error, fp_t kp, fp_t ki, fp_t kd,
                     fp_t integral_limit, fp_t d_alpha, fp_t dt,
                     const PIDState* prev, PIDState* next) {
    fp_t integral = prev->integral + FP_MUL(error, dt);
    integral = FP_CLIP(integral, -integral_limit, integral_limit);
    next->integral = integral;

    fp_t derivative = FP_DIV(error - prev->error_prev, dt);
    fp_t d_filtered = FP_MUL(d_alpha, derivative) + FP_MUL(FP_ONE - d_alpha, prev->d_filtered);
    next->d_filtered = d_filtered;
    next->error_prev = error;

    fp_t p_term = FP_MUL(kp, error);
    fp_t i_term = FP_MUL(ki, integral);
    fp_t d_term = FP_MUL(kd, d_filtered);

    return p_term + i_term + d_term;
}

// === Phase 2: Cascade PID ===

PIDResult pid_cascade(fp_t v_target, fp_t v_actual, fp_t a_actual_est,
                      fp_t kp_outer, fp_t ki_outer, fp_t kd_outer,
                      fp_t kp_inner, fp_t ki_inner,
                      fp_t integral_limit_outer, fp_t integral_limit_inner,
                      fp_t d_alpha, fp_t jerk_max,
                      fp_t brake_ramp_rate, fp_t brake_prev,
                      const PIDState* prev_outer, const PIDState* prev_inner,
                      fp_t dt) {
    PIDResult result;

    fp_t v_error = v_target - v_actual;
    fp_t a_target = pid_step(v_error, kp_outer, ki_outer, kd_outer,
                             integral_limit_outer, d_alpha, dt,
                             prev_outer, &result.outer);

    a_target = FP_CLIP(a_target, FP_FROM_FLOAT(-5.0f), FP_FROM_FLOAT(3.0f));

    if (jerk_max > 0 && dt > 0) {
        fp_t max_a_delta = FP_MUL(jerk_max, dt);
        fp_t a_prev = prev_outer->a_prev;
        fp_t delta_a = a_target - a_prev;
        if (delta_a > max_a_delta) delta_a = max_a_delta;
        if (delta_a < -max_a_delta) delta_a = -max_a_delta;
        a_target = a_prev + delta_a;
    }

    result.a_target = a_target;
    result.outer.a_prev = a_target;

    fp_t a_error = a_target - a_actual_est;
    fp_t accel_cmd = pid_step(a_error, kp_inner, ki_inner, FP_FROM_INT(0),
                              integral_limit_inner, FP_FROM_INT(0), dt,
                              prev_inner, &result.inner);

    fp_t throttle;
    fp_t brake;
    if (accel_cmd >= 0) {
        throttle = FP_CLIP(accel_cmd, 0, FP_ONE);
        brake = 0;
    } else {
        throttle = 0;
        brake = FP_CLIP(-accel_cmd, 0, FP_ONE);
    }

    if (brake_ramp_rate > 0 && dt > 0) {
        fp_t max_brake_inc = FP_MUL(brake_ramp_rate, dt);
        fp_t brake_delta = brake - brake_prev;
        if (brake_delta > max_brake_inc) brake_delta = max_brake_inc;
        fp_t brake_release = max_brake_inc * 3;
        if (brake_delta < -brake_release) brake_delta = -brake_release;
        brake = brake_prev + brake_delta;
    }

    result.throttle = throttle;
    result.brake = brake;

    return result;
}
