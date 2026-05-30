#include "localization.h"
#include "lookup_table.h"
#include <string.h>

// === Phase 1: Matrix Helpers ===

static fp_t mat_get(const fp_t* m, int row, int col, int n) {
    return m[row * n + col];
}

static void mat_set(fp_t* m, int row, int col, int n, fp_t val) {
    m[row * n + col] = val;
}

static void mat_identity(fp_t* m, int n) {
    memset(m, 0, n * n * sizeof(fp_t));
    for (int i = 0; i < n; i++) m[i * n + i] = FP_ONE;
}

// === Phase 2: EKF Predict ===

void ekf_predict_fp(EKFState* state, EKFCovariance* P,
                    fp_t v_cmd, fp_t yaw_rate_cmd,
                    const EKFParams* params, fp_t dt) {
    fp_t cos_yaw = lut_cos((int16_t)FP_TO_INT(FP_MUL(state->yaw, FP_RAD_TO_DEG)));
    fp_t sin_yaw = lut_sin((int16_t)FP_TO_INT(FP_MUL(state->yaw, FP_RAD_TO_DEG)));

    state->x += FP_MUL(v_cmd, FP_MUL(cos_yaw, dt));
    state->y += FP_MUL(v_cmd, FP_MUL(sin_yaw, dt));
    state->yaw += FP_MUL(yaw_rate_cmd, dt);
    state->v = v_cmd;
    if (state->v < 0) state->v = 0;

    fp_t accel_factor = FP_ONE + FP_MUL(params->accel_scale, FP_ABS(v_cmd));
    fp_t curvature_factor = FP_ONE + FP_MUL(FP_FROM_FLOAT(3.0f), FP_ABS(yaw_rate_cmd));
    fp_t combined_factor = FP_MUL(accel_factor, curvature_factor);
    fp_t q_x = FP_MUL(params->q_x, combined_factor);
    fp_t q_y = FP_MUL(params->q_y, combined_factor);
    fp_t q_yaw = FP_MUL(params->q_yaw, combined_factor);
    fp_t q_v = FP_MUL(params->q_v, combined_factor);

    fp_t* p = P->data;
    fp_t p_max = FP_FROM_FLOAT(100.0f);
    mat_set(p, 0, 0, EKF_STATE_SIZE, FP_CLIP(mat_get(p, 0, 0, EKF_STATE_SIZE) + q_x, -p_max, p_max));
    mat_set(p, 1, 1, EKF_STATE_SIZE, FP_CLIP(mat_get(p, 1, 1, EKF_STATE_SIZE) + q_y, -p_max, p_max));
    mat_set(p, 2, 2, EKF_STATE_SIZE, FP_CLIP(mat_get(p, 2, 2, EKF_STATE_SIZE) + q_yaw, -p_max, p_max));
    mat_set(p, 3, 3, EKF_STATE_SIZE, FP_CLIP(mat_get(p, 3, 3, EKF_STATE_SIZE) + q_v, -p_max, p_max));
}

// === Phase 3: EKF Update ===

void ekf_update_fp(EKFState* state, EKFCovariance* P,
                   fp_t z_x, fp_t z_y,
                   const EKFParams* params) {
    fp_t y_x = z_x - state->x;
    fp_t y_y = z_y - state->y;

    fp_t* p = P->data;
    int n = EKF_STATE_SIZE;

    fp_t S00 = mat_get(p, 0, 0, n) + params->r_x;
    fp_t S01 = mat_get(p, 0, 1, n);
    fp_t S10 = mat_get(p, 1, 0, n);
    fp_t S11 = mat_get(p, 1, 1, n) + params->r_y;

    fp_t det_S = FP_MUL(S00, S11) - FP_MUL(S01, S10);
    if (FP_ABS(det_S) < FP_EPSILON) return;

    fp_t inv_det = FP_DIV(FP_ONE, det_S);
    fp_t Si00 = FP_MUL(S11, inv_det);
    fp_t Si01 = FP_MUL(-S01, inv_det);
    fp_t Si10 = FP_MUL(-S10, inv_det);
    fp_t Si11 = FP_MUL(S00, inv_det);

    fp_t K00 = FP_MUL(mat_get(p, 0, 0, n), Si00) + FP_MUL(mat_get(p, 0, 1, n), Si10);
    fp_t K01 = FP_MUL(mat_get(p, 0, 0, n), Si01) + FP_MUL(mat_get(p, 0, 1, n), Si11);
    fp_t K10 = FP_MUL(mat_get(p, 1, 0, n), Si00) + FP_MUL(mat_get(p, 1, 1, n), Si10);
    fp_t K11 = FP_MUL(mat_get(p, 1, 0, n), Si01) + FP_MUL(mat_get(p, 1, 1, n), Si11);
    fp_t K20 = FP_MUL(mat_get(p, 2, 0, n), Si00) + FP_MUL(mat_get(p, 2, 1, n), Si10);
    fp_t K21 = FP_MUL(mat_get(p, 2, 0, n), Si01) + FP_MUL(mat_get(p, 2, 1, n), Si11);
    fp_t K30 = FP_MUL(mat_get(p, 3, 0, n), Si00) + FP_MUL(mat_get(p, 3, 1, n), Si10);
    fp_t K31 = FP_MUL(mat_get(p, 3, 0, n), Si01) + FP_MUL(mat_get(p, 3, 1, n), Si11);

    state->x += FP_MUL(K00, y_x) + FP_MUL(K01, y_y);
    state->y += FP_MUL(K10, y_x) + FP_MUL(K11, y_y);
    state->yaw += FP_MUL(K20, y_x) + FP_MUL(K21, y_y);
    state->v += FP_MUL(K30, y_x) + FP_MUL(K31, y_y);
    if (state->v < 0) state->v = 0;

    fp_t KH[EKF_STATE_SIZE * EKF_STATE_SIZE];
    memset(KH, 0, sizeof(KH));
    KH[0 * n + 0] = FP_MUL(K00, FP_ONE);
    KH[0 * n + 1] = FP_MUL(K01, FP_ONE);
    KH[1 * n + 0] = FP_MUL(K10, FP_ONE);
    KH[1 * n + 1] = FP_MUL(K11, FP_ONE);
    KH[2 * n + 0] = FP_MUL(K20, FP_ONE);
    KH[2 * n + 1] = FP_MUL(K21, FP_ONE);
    KH[3 * n + 0] = FP_MUL(K30, FP_ONE);
    KH[3 * n + 1] = FP_MUL(K31, FP_ONE);

    fp_t I_KH[EKF_STATE_SIZE * EKF_STATE_SIZE];
    mat_identity(I_KH, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            I_KH[i * n + j] -= KH[i * n + j];

    fp_t P_new[EKF_STATE_SIZE * EKF_STATE_SIZE];
    memset(P_new, 0, sizeof(P_new));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                P_new[i * n + j] += FP_MUL(I_KH[i * n + k], mat_get(p, k, j, n));

    fp_t IKHP[EKF_STATE_SIZE * EKF_STATE_SIZE];
    memset(IKHP, 0, sizeof(IKHP));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                IKHP[i * n + j] += FP_MUL(P_new[i * n + k], I_KH[j * n + k]);

    fp_t KR[EKF_STATE_SIZE * 2];
    memset(KR, 0, sizeof(KR));
    KR[0 * 2 + 0] = FP_MUL(K00, params->r_x);
    KR[0 * 2 + 1] = FP_MUL(K01, params->r_y);
    KR[1 * 2 + 0] = FP_MUL(K10, params->r_x);
    KR[1 * 2 + 1] = FP_MUL(K11, params->r_y);
    KR[2 * 2 + 0] = FP_MUL(K20, params->r_x);
    KR[2 * 2 + 1] = FP_MUL(K21, params->r_y);
    KR[3 * 2 + 0] = FP_MUL(K30, params->r_x);
    KR[3 * 2 + 1] = FP_MUL(K31, params->r_y);

    fp_t K_mat[EKF_STATE_SIZE * 2];
    K_mat[0 * 2 + 0] = K00; K_mat[0 * 2 + 1] = K01;
    K_mat[1 * 2 + 0] = K10; K_mat[1 * 2 + 1] = K11;
    K_mat[2 * 2 + 0] = K20; K_mat[2 * 2 + 1] = K21;
    K_mat[3 * 2 + 0] = K30; K_mat[3 * 2 + 1] = K31;

    fp_t KRKT[EKF_STATE_SIZE * EKF_STATE_SIZE];
    memset(KRKT, 0, sizeof(KRKT));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < 2; k++)
                KRKT[i * n + j] += FP_MUL(KR[i * 2 + k], K_mat[j * 2 + k]);

    for (int i = 0; i < n * n; i++)
        P_new[i] = IKHP[i] + KRKT[i];

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            fp_t avg = (P_new[i * n + j] + P_new[j * n + i]) >> 1;
            P_new[i * n + j] = avg;
            P_new[j * n + i] = avg;
        }

    memcpy(P->data, P_new, sizeof(P_new));
}
