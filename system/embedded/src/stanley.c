#include "stanley.h"
#include "lookup_table.h"

// === Phase 1: Nearest Path Point ===

static uint16_t find_nearest(const PathBuffer* path, fp_t wheelbase) {
    fp_t dist_min = FP_FROM_INT(0x7FFF);
    uint16_t idx_min = 0;
    for (uint16_t i = 0; i < path->n; i++) {
        fp_t dx = path->x[i] - wheelbase;
        fp_t dy = path->y[i];
        fp_t dist = FP_MUL(dx, dx) + FP_MUL(dy, dy);
        if (dist < dist_min) {
            dist_min = dist;
            idx_min = i;
        }
    }
    return idx_min;
}

// === Phase 2: Preview Index ===

static uint16_t find_preview_idx(const PathBuffer* path, uint16_t nearest_idx,
                                  fp_t preview_dist) {
    fp_t s_cum[STANLEY_MAX_PATH];
    s_cum[0] = 0;
    for (uint16_t i = 1; i < path->n; i++) {
        fp_t dx = path->x[i] - path->x[i - 1];
        fp_t dy = path->y[i] - path->y[i - 1];
        fp_t ds = fp_sqrt(FP_MUL(dx, dx) + FP_MUL(dy, dy));
        s_cum[i] = s_cum[i - 1] + ds;
    }
    fp_t s_target = s_cum[nearest_idx] + preview_dist;
    uint16_t idx = nearest_idx;
    for (uint16_t i = nearest_idx + 1; i < path->n; i++) {
        if (s_cum[i] >= s_target) {
            idx = i;
            break;
        }
        idx = i;
    }
    return idx;
}

// === Phase 3: Q16.16 Radians To Degrees ===

static int16_t rad_to_deg_int(fp_t rad) {
    fp_t deg_q16 = FP_MUL(rad, FP_RAD_TO_DEG);
    int16_t deg = (int16_t)FP_TO_INT(deg_q16);
    return deg;
}

// === Phase 4: Stanley Steer ===

StanleyResult stanley_steer(const PathBuffer* path, fp_t speed, fp_t k_e, fp_t k_v,
                            fp_t wheelbase, fp_t k_preview,
                            fp_t steer_prev, fp_t max_steer_rate_rad_per_s,
                            fp_t dead_zone, fp_t v_damping,
                            fp_t max_steer_rad, fp_t dt) {
    StanleyResult result;
    result.nearest_idx = find_nearest(path, wheelbase);
    result.preview_idx = result.nearest_idx;

    fp_t speed_safe = speed;
    if (speed_safe < FP_FROM_FLOAT(0.5f)) speed_safe = FP_FROM_FLOAT(0.5f);
    fp_t preview_dist = FP_MUL(k_preview, speed_safe);
    if (preview_dist > 0) {
        result.preview_idx = find_preview_idx(path, result.nearest_idx, preview_dist);
    }

    uint16_t idx_preview = result.preview_idx;
    result.e_heading = path->theta[idx_preview];

    uint16_t idx_near = result.nearest_idx;
    int16_t theta_deg = rad_to_deg_int(path->theta[idx_near]);
    fp_t sin_theta = lut_sin(theta_deg);
    fp_t cos_theta = lut_cos(theta_deg);

    fp_t dx_front = path->x[idx_near] - wheelbase;
    fp_t dy_front = path->y[idx_near];
    fp_t e_lat_front = -FP_MUL(dx_front, sin_theta) + FP_MUL(dy_front, cos_theta);

    result.e_lat = -FP_MUL(path->x[idx_near], sin_theta) + FP_MUL(path->y[idx_near], cos_theta);

    fp_t e_lat_eff = e_lat_front;
    if (FP_ABS(e_lat_eff) < dead_zone) e_lat_eff = 0;

    fp_t k_e_eff = k_e;
    if (v_damping > 0) {
        fp_t ratio = FP_DIV(speed, v_damping);
        if (ratio < FP_ONE) {
            k_e_eff = FP_MUL(k_e, ratio);
        }
    }

    fp_t k_e_e_lat = FP_MUL(k_e_eff, e_lat_eff);
    fp_t speed_adj = speed + k_v;
    if (speed_adj < FP_EPSILON) speed_adj = FP_EPSILON;

    fp_t cross_term = lut_atan2(k_e_e_lat, speed_adj);

    fp_t steer_fb = result.e_heading + cross_term;

    fp_t kappa_near = path->kappa[idx_near];
    fp_t steer_ff = lut_atan2(FP_MUL(kappa_near, wheelbase), FP_ONE);

    fp_t steer = steer_fb + steer_ff;

    if (steer_prev != 0 && dt > 0 && max_steer_rate_rad_per_s > 0) {
        fp_t max_delta = FP_MUL(max_steer_rate_rad_per_s, dt);
        fp_t delta = steer - steer_prev;
        if (delta > max_delta) delta = max_delta;
        if (delta < -max_delta) delta = -max_delta;
        steer = steer_prev + delta;
    }

    steer = FP_CLIP(steer, -max_steer_rad, max_steer_rad);

    result.steer_rad = steer;
    return result;
}
