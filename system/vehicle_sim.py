import numpy as np

WHEELBASE = 2.7

def simulate_vehicle(steer_deg, throttle, brake, v_actual,
                     x_actual, y_actual, theta_actual, dt,
                     wheelbase=WHEELBASE, throttle_gain=3.0, brake_gain=5.0):
    steer_rad = np.radians(steer_deg)
    accel = throttle * throttle_gain - brake * brake_gain
    v_new = max(v_actual + accel * dt, 0.0)
    v_mid = 0.5 * (v_actual + v_new)
    if abs(steer_rad) > 1e-6:
        kappa = np.tan(steer_rad) / wheelbase
        theta_new = theta_actual + kappa * v_mid * dt
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        x_new = x_actual + v_mid * np.cos(theta_actual + kappa * v_actual * dt * 0.5) * dt
        y_new = y_actual + v_mid * np.sin(theta_actual + kappa * v_actual * dt * 0.5) * dt
    else:
        theta_new = theta_actual
        x_new = x_actual + v_mid * np.cos(theta_actual) * dt
        y_new = y_actual + v_mid * np.sin(theta_actual) * dt
    return v_new, x_new, y_new, theta_new
