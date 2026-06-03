import sys
import os
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from PathTracking.pure_pursuit_controller import pure_pursuit_control
from PathTracking.stanley_controller import stanley_control
from PathTracking.fuzzy_controller import fuzzy_control
from PathTracking.mpc_controller import mpc_control
from PathTracking.rl_controller import rl_control

CTRL_PURE_PURSUIT = 0
CTRL_STANLEY = 1
CTRL_FUZZY = 2
CTRL_MPC = 3
CTRL_RL = 4

_CTRL_NAMES = {
    CTRL_PURE_PURSUIT: "PurePursuit",
    CTRL_STANLEY: "Stanley",
    CTRL_FUZZY: "Fuzzy",
    CTRL_MPC: "MPC",
    CTRL_RL: "RL",
}


def select_controller(ctrl_type, decision_output, speed_actual, state,
                      vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0, dt=0.02):
    speed_actual = max(float(speed_actual) if np.isfinite(speed_actual) else 0.0, 0.0)
    prev_type = state.get("type", None) if state else None
    prev_steer = state.get("prev_steer", 0.0) if state else 0.0
    if ctrl_type != prev_type:
        state = {"prev_steer": prev_steer, "bumpless_steer": prev_steer}
    if ctrl_type == CTRL_PURE_PURSUIT:
        pid_state = state.get("pid", None) if state else None
        if pid_state is None:
            pid_state = (0.0, 0.0)
        ctrl_out, new_pid = pure_pursuit_control(
            decision_output, speed_actual=speed_actual,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_theta=vehicle_theta,
            pid_state=pid_state, dt=dt
        )
        new_state = {"pid": new_pid, "type": ctrl_type, "prev_steer": float(ctrl_out.steering.steering_angle)}
        return ctrl_out, new_state

    elif ctrl_type == CTRL_STANLEY:
        stanley_state = state.get("stanley", None) if state else None
        if stanley_state is None and "bumpless_steer" in (state or {}):
            stanley_state = {"outer": (0.0, 0.0), "inner": 0.0, "steer_prev": None,
                             "d_filtered": 0.0, "a_prev": 0.0, "brake_prev": 0.0}
        ctrl_out, new_stanley = stanley_control(
            decision_output, speed_actual=speed_actual,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_theta=vehicle_theta,
            pid_state=stanley_state, dt=dt
        )
        new_state = {"stanley": new_stanley, "type": ctrl_type, "prev_steer": float(ctrl_out.steering.steering_angle)}
        return ctrl_out, new_state

    elif ctrl_type == CTRL_FUZZY:
        steer_state = state.get("steer", None) if state else None
        fuzzy_state = state.get("fuzzy", None) if state else None
        if steer_state is None and "bumpless_steer" in (state or {}):
            steer_state = {"prev_steer": state["bumpless_steer"]}
        ctrl_out, new_inner = fuzzy_control(
            decision_output, speed_actual=speed_actual,
            steer_state=steer_state, fuzzy_state=fuzzy_state, dt=dt,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_theta=vehicle_theta
        )
        new_state = {"steer": new_inner["steer"], "fuzzy": new_inner["fuzzy"], "type": ctrl_type, "prev_steer": float(ctrl_out.steering.steering_angle)}
        return ctrl_out, new_state

    elif ctrl_type == CTRL_MPC:
        mpc_state = state.get("mpc", None) if state else None
        ctrl_out, new_mpc = mpc_control(
            decision_output, speed_actual=speed_actual, mpc_state=mpc_state, dt=dt,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_theta=vehicle_theta
        )
        new_state = {"mpc": new_mpc, "type": ctrl_type, "prev_steer": float(ctrl_out.steering.steering_angle)}
        return ctrl_out, new_state

    elif ctrl_type == CTRL_RL:
        weights = state.get("weights", None) if state else None
        rl_state = state.get("rl", None) if state else None
        ctrl_out, new_rl = rl_control(
            decision_output, speed_actual=speed_actual,
            weights=weights, rl_state=rl_state, dt=dt,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_theta=vehicle_theta
        )
        new_state = {"rl": new_rl, "weights": weights, "type": ctrl_type, "prev_steer": float(ctrl_out.steering.steering_angle)}
        return ctrl_out, new_state

    else:
        stanley_state = state.get("stanley", None) if state else None
        ctrl_out, new_stanley = stanley_control(
            decision_output, speed_actual=speed_actual,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_theta=vehicle_theta,
            pid_state=stanley_state, dt=dt
        )
        new_state = {"stanley": new_stanley, "type": CTRL_STANLEY, "prev_steer": float(ctrl_out.steering.steering_angle)}
        return ctrl_out, new_state


def ctrl_name(ctrl_type):
    return _CTRL_NAMES.get(ctrl_type, "Unknown")


def auto_select_controller(decision_output, speed_actual, state,
                           vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0, dt=0.02):
    n_path = len(decision_output.target_path)
    if n_path < 1:
        return select_controller(CTRL_STANLEY, decision_output, speed_actual, state,
                                 vehicle_x, vehicle_y, vehicle_theta, dt)

    path_x = np.array([p.pose.x for p in decision_output.target_path])
    path_y = np.array([p.pose.y for p in decision_output.target_path])
    dx = path_x - vehicle_x
    dy = path_y - vehicle_y
    idx_nearest = int(np.argmin(dx**2 + dy**2))

    kappa_arr = np.array([decision_output.target_path[i].curvature
                          for i in range(idx_nearest, min(idx_nearest + 20, n_path))])
    kappa_max = float(np.abs(kappa_arr).max()) if len(kappa_arr) > 0 else 0.0

    lookahead = min(idx_nearest + 10, n_path - 1)
    dist_to_end = np.sqrt((path_x[lookahead] - path_x[-1])**2 +
                          (path_y[lookahead] - path_y[-1])**2)

    if speed_actual < 1.0:
        ctrl_type = CTRL_STANLEY
    elif kappa_max > 0.15:
        ctrl_type = CTRL_MPC
    elif kappa_max > 0.03:
        ctrl_type = CTRL_PURE_PURSUIT
    elif speed_actual > 8.0:
        ctrl_type = CTRL_PURE_PURSUIT
    elif dist_to_end < 5.0:
        ctrl_type = CTRL_STANLEY
    else:
        ctrl_type = CTRL_PURE_PURSUIT

    return select_controller(ctrl_type, decision_output, speed_actual, state,
                             vehicle_x, vehicle_y, vehicle_theta, dt)
