#include "slam_config.h"
#include "localization.h"
#include "lookup_table.h"
#include <string.h>

// === Phase 1: SLAM Init ===

void slam_init(SLAMState* state, fp_t x0, fp_t y0, fp_t yaw0,
               uint16_t n_particles, uint16_t n_landmarks) {
    state->n_particles = n_particles > SLAM_MAX_PARTICLES ? SLAM_MAX_PARTICLES : n_particles;
    state->n_landmarks = n_landmarks > SLAM_MAX_LANDMARKS ? SLAM_MAX_LANDMARKS : n_landmarks;
    state->traj_x = x0;
    state->traj_y = y0;
    state->traj_yaw = yaw0;

    fp_t w_init = FP_DIV(FP_ONE, FP_FROM_INT(state->n_particles));

    for (uint16_t i = 0; i < state->n_particles; i++) {
        state->particles[i].x = x0;
        state->particles[i].y = y0;
        state->particles[i].yaw = yaw0;
        state->particles[i].w = w_init;
        for (uint16_t j = 0; j < state->n_landmarks; j++) {
            state->particles[i].lm[j].x = 0;
            state->particles[i].lm[j].y = 0;
            state->particles[i].lm[j].observed = 0;
        }
        memset(state->particles[i].lmP, 0, sizeof(state->particles[i].lmP));
    }
}

// === Phase 2: SLAM Predict ===

void slam_predict(SLAMState* state, fp_t v_cmd, fp_t yaw_rate_cmd, fp_t dt) {
    for (uint16_t i = 0; i < state->n_particles; i++) {
        fp_t cos_yaw = lut_cos((int16_t)FP_TO_INT(FP_MUL(state->particles[i].yaw, FP_RAD_TO_DEG)));
        fp_t sin_yaw = lut_sin((int16_t)FP_TO_INT(FP_MUL(state->particles[i].yaw, FP_RAD_TO_DEG)));
        state->particles[i].x += FP_MUL(v_cmd, FP_MUL(cos_yaw, dt));
        state->particles[i].y += FP_MUL(v_cmd, FP_MUL(sin_yaw, dt));
        state->particles[i].yaw += FP_MUL(yaw_rate_cmd, dt);
    }
}

// === Phase 3: SLAM Estimate ===

void slam_estimate(const SLAMState* state, EKFState* est) {
    fp_t x_sum = 0, y_sum = 0, sin_sum = 0, cos_sum = 0;
    for (uint16_t i = 0; i < state->n_particles; i++) {
        x_sum += FP_MUL(state->particles[i].x, state->particles[i].w);
        y_sum += FP_MUL(state->particles[i].y, state->particles[i].w);
        sin_sum += FP_MUL(lut_sin((int16_t)FP_TO_INT(FP_MUL(state->particles[i].yaw, FP_RAD_TO_DEG))), state->particles[i].w);
        cos_sum += FP_MUL(lut_cos((int16_t)FP_TO_INT(FP_MUL(state->particles[i].yaw, FP_RAD_TO_DEG))), state->particles[i].w);
    }
    est->x = x_sum;
    est->y = y_sum;
    est->yaw = lut_atan2(sin_sum, cos_sum);
    est->v = 0;
}

// === Phase 4: ICP Match Fixed-Point ===

void icp_match_fp(const fp_t* prev_pts, const fp_t* curr_pts,
                  uint16_t n_pts, ICPResultFP* result) {
    result->R[0] = FP_ONE;
    result->R[1] = 0;
    result->R[2] = 0;
    result->R[3] = FP_ONE;
    result->T[0] = 0;
    result->T[1] = 0;
    result->final_error = 0;
    result->iterations = 0;

    if (n_pts < 3) return;

    fp_t prev_err = 0;
    for (uint16_t iter = 0; iter < SLAM_ICP_MAX_ITER; iter++) {
        result->iterations = iter + 1;

        fp_t err_sum = 0;
        fp_t pm_x = 0, pm_y = 0, cm_x = 0, cm_y = 0;

        for (uint16_t i = 0; i < n_pts; i++) {
            fp_t min_d2 = 0x7FFFFFFF;
            for (uint16_t j = 0; j < n_pts; j++) {
                fp_t dx = curr_pts[i * 2] - prev_pts[j * 2];
                fp_t dy = curr_pts[i * 2 + 1] - prev_pts[j * 2 + 1];
                fp_t d2 = FP_MUL(dx, dx) + FP_MUL(dy, dy);
                if (d2 < min_d2) min_d2 = d2;
            }
            err_sum += min_d2;
            pm_x += prev_pts[i * 2];
            pm_y += prev_pts[i * 2 + 1];
            cm_x += curr_pts[i * 2];
            cm_y += curr_pts[i * 2 + 1];
        }

        fp_t err = fp_sqrt(FP_DIV(err_sum, FP_FROM_INT(n_pts)));
        result->final_error = err;

        if (iter > 0 && FP_ABS(prev_err - err) < SLAM_ICP_EPS) break;
        prev_err = err;

        pm_x = FP_DIV(pm_x, FP_FROM_INT(n_pts));
        pm_y = FP_DIV(pm_y, FP_FROM_INT(n_pts));
        cm_x = FP_DIV(cm_x, FP_FROM_INT(n_pts));
        cm_y = FP_DIV(cm_y, FP_FROM_INT(n_pts));

        result->T[0] = pm_x - cm_x;
        result->T[1] = pm_y - cm_y;
    }
}
