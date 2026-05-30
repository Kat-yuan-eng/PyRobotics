import numpy as np


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rot_mat2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_to_global(x_veh, y_veh, vehicle_x, vehicle_y, vehicle_theta):
    R = rot_mat2d(vehicle_theta)
    pt_veh = np.array([x_veh, y_veh])
    pt_global = R @ pt_veh + np.array([vehicle_x, vehicle_y])
    return pt_global[0], pt_global[1]


def transform_to_vehicle(x_global, y_global, vehicle_x, vehicle_y, vehicle_theta):
    dx = x_global - vehicle_x
    dy = y_global - vehicle_y
    cos_v = np.cos(vehicle_theta)
    sin_v = np.sin(vehicle_theta)
    x_veh = dx * cos_v + dy * sin_v
    y_veh = -dx * sin_v + dy * cos_v
    return x_veh, y_veh
