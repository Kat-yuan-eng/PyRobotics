import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw, length, width, fc, ec)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  head_length=width * 0.8, head_width=width, fc=fc, ec=ec)


def plot_vehicle(x, y, yaw, length=4.5, width=2.0, color="blue"):
    corners = np.array([
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
        [length / 2, width / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated = (R @ corners.T).T + np.array([x, y])
    plt.fill(rotated[:, 0], rotated[:, 1], color=color, alpha=0.3)
    plt.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=1.5)


def generate_serpentine_course(ds=0.1):
    ax = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    ay = [0.0, 20.0, -20.0, 20.0, -20.0, 20.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_course(ax, ay, ds=ds)
    return cx, cy, cyaw, ck


def generate_circle_course(radius=30.0, ds=0.1):
    theta = np.arange(0, 2 * np.pi, ds / radius)
    cx = radius * np.cos(theta)
    cy = radius * np.sin(theta)
    cyaw = np.arctan2(np.gradient(cy), np.gradient(cx))
    ck = np.gradient(cyaw) / ds
    return cx, cy, cyaw, ck


def cubic_spline_course(ax, ay, ds=0.1):
    n = len(ax)
    t = np.arange(n, dtype=float)
    t_fine = np.arange(0, n - 1, ds / max(1.0, np.max(np.sqrt(np.diff(ax)**2 + np.diff(ay)**2))))
    cs_x = CubicSpline(t, ax)
    cs_y = CubicSpline(t, ay)
    cx = cs_x(t_fine)
    cy = cs_y(t_fine)
    dx = cs_x(t_fine, 1)
    dy = cs_y(t_fine, 1)
    ddx = cs_x(t_fine, 2)
    ddy = cs_y(t_fine, 2)
    cyaw = np.arctan2(dy, dx)
    ck = (ddy * dx - ddx * dy) / ((dx**2 + dy**2)**1.5 + 1e-12)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
    return cx, cy, cyaw, ck, s
