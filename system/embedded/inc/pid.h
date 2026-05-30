#ifndef PID_H
#define PID_H

#include "fixed_point.h"

// === Phase 1: PID State ===

typedef struct {
    fp_t integral;
    fp_t error_prev;
    fp_t d_filtered;
    fp_t a_prev;
} PIDState;

typedef struct {
    PIDState outer;
    PIDState inner;
    fp_t throttle;
    fp_t brake;
    fp_t a_target;
} PIDResult;

// === Phase 2: Cascade PID ===

PIDResult pid_cascade(fp_t v_target, fp_t v_actual, fp_t a_actual_est,
                      fp_t kp_outer, fp_t ki_outer, fp_t kd_outer,
                      fp_t kp_inner, fp_t ki_inner,
                      fp_t integral_limit_outer, fp_t integral_limit_inner,
                      fp_t d_alpha, fp_t jerk_max,
                      fp_t brake_ramp_rate, fp_t brake_prev,
                      const PIDState* prev_outer, const PIDState* prev_inner,
                      fp_t dt);

#endif
