# PyRobotics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](CHANGELOG.md)

Intelligent vehicle autonomous driving software stack implementing a four-layer architecture: **Perception → Decision → Control → System**, with all modules decoupled via Protobuf messages.

[中文文档](README_CN.md)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PyRobotics System                         │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│Perception│ Decision │ Control  │Localization│     SLAM      │
├──────────┼──────────┼──────────┼──────────┼────────────────┤
│Lane Det. │Scheduler │Stanley   │EKF       │FastSLAM 2.0   │
│Obst. Det.│Smoother  │PurePurs. │PF        │ICP Matching   │
│Sign Rec. │Avoidance │Fuzzy     │Fusion    │SLAM Pipeline  │
│Sensor Fus│MultiAgent│MPC / DQN │          │               │
├──────────┴──────────┴──────────┴──────────┴────────────────┤
│              System Infrastructure                           │
│  Vehicle Sim │ Realtime Pipeline │ Embedded C │ ROS2        │
└─────────────────────────────────────────────────────────────┘
```

## Module Overview

| Subsystem | Directory | Core Function | Key Algorithms |
|-----------|-----------|---------------|----------------|
| Perception | `perception/` | Lane/obstacle/sign detection & fusion | HLS color seg, RANSAC+DBSCAN, HOG template match |
| Decision | `decision/` | Task scheduling, path smoothing, avoidance | FSM dispatch, curvature-constrained smooth, lateral offset |
| Planning | `PathPlanning/` | Global planning & local avoidance | A\*, RRT (with smoothing), DWA (global path alignment) |
| Control | `PathTracking/` | Lateral + longitudinal vehicle control | Stanley, Pure Pursuit, Fuzzy, MPC, DQN, Selector |
| Localization | `Localization/` | Vehicle pose estimation | EKF (Joseph-form), Particle Filter, Covariance Intersection |
| SLAM | `SLAM/` | Simultaneous localization & mapping | FastSLAM 2.0 (adaptive threshold), ICP (Huber kernel) |
| System | `system/` | Vehicle sim & realtime framework | Bicycle model, dual-thread pipeline, Embedded C, ROS2 |

## Features

### Perception (5 modules)
- **Lane Pixel Detector**: Scan-line peak detection with jump filter
- **Obstacle Detector**: RANSAC ground removal + DBSCAN clustering + OBB fitting
- **Obstacle Tracker**: Hungarian assignment + Kalman velocity estimation
- **Sign Recognizer**: HOG descriptor + template matching
- **Sensor Fusion**: Pinhole model back-projection to vehicle coordinates

### Decision (4 modules)
- **Task Scheduler**: FSM (PATROL → AVOID → PARK) with Protobuf output
- **Path Smoother**: Curvature-constrained smoothing with deviation bounds
- **Obstacle Avoidance**: Lateral offset path generation
- **Multi-Agent**: ID-offset coordination with velocity validation

### Path Planning (3 algorithms)
- **A\***: Grid-based optimal search
- **RRT**: Sampling-based with integrated path smoothing
- **DWA**: Dynamic Window Approach with global path alignment cost

### Path Tracking (5 controllers + selector)
- **Stanley**: Front-axle lateral error feedback
- **Pure Pursuit**: Lookahead-based geometric control
- **Fuzzy**: Curvature-adaptive with complete membership overlap
- **MPC**: Model Predictive Control with terminal cost
- **DQN**: Deep Q-Network with EMA smoothing and safety filter
- **Controller Selector**: Speed/curvature-based switching with Bumpless Transfer

### Localization (3 algorithms)
- **EKF**: Extended Kalman Filter with Joseph-form covariance update
- **Particle Filter**: Systematic resampling with NaN-safe weights
- **Fusion**: Covariance Intersection with divergence detection

### SLAM (3 modules)
- **FastSLAM 2.0**: Adaptive resampling threshold, Joseph-form landmark update
- **ICP Matching**: Initial pose guess + Huber robust kernel
- **SLAM Pipeline**: Integrated EKF + FastSLAM with ICP refinement

### System Infrastructure
- **Vehicle Simulator**: Bicycle kinematic model
- **Realtime Pipeline**: Dual-thread (perception 30ms + control 50Hz)
- **Embedded C**: PID, Stanley, EKF, vehicle control with Q16.16 fixed-point
- **ROS2**: Perception, planning, control nodes with ApproximateTimeSynchronizer
- **Protobuf**: 6 message definitions (common, perception, decision, control, system, agent)

## Safety Features

- Throttle/brake mutual exclusion (`if brake > 0.01: throttle = 0.0`)
- EKF covariance explosion detection (reset when max(diag) > 100)
- NaN guards in all estimators (fallback to prediction)
- Obstacle coordinate validation (finite check)
- ID collision prevention in multi-agent and tracker
- Joseph-form covariance update (guarantees positive definiteness)

## Quick Start

### Requirements

- Python >= 3.9
- NumPy, SciPy, Matplotlib
- OpenCV (cv2), scikit-learn
- protobuf + grpcio-tools
- pytest

### Installation

```bash
cd PyRobotics
pip install -r requirements/requirements.txt

# Regenerate Protobuf code (if needed)
python -m grpc_tools.protoc -I=proto --python_out=generated proto/*.proto
```

### Usage

#### Single Module Visualization

Each module can be run independently with PythonRobotics-style interactive animation:

```bash
python PathTracking/stanley_controller.py
python PathPlanning/dwa_planner.py
python Localization/ekf_localizer.py
python SLAM/fast_slam.py
```

#### Main Loop Simulation

```bash
python main_loop.py
```

Single-threaded 50Hz closed-loop: Perception → Decision → Control → Vehicle Sim.

#### Unit Tests

```bash
python tests/test_all_modules.py
```

## Project Structure

```
PyRobotics/
├── perception/          # Lane/obstacle/sign detection & sensor fusion
├── decision/            # Task scheduling, path smoothing, avoidance
├── PathTracking/        # 5 tracking controllers + selector
├── PathPlanning/        # A*, RRT, DWA planners + selector
├── Localization/        # EKF, PF, Fusion localizers
├── SLAM/                # FastSLAM 2.0, ICP, pipeline
├── system/              # Vehicle sim, realtime, embedded C, ROS2
├── utils/               # Plot, angle, geometry helpers
├── generated/           # Protobuf generated Python code
├── proto/               # Protobuf message definitions
├── tests/               # Unit tests
├── docs/                # Architecture, API reference, quick start
└── requirements/        # Dependency list
```

## Acknowledgments

This project is inspired by [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) by Atsushi Sakai et al., which provides an excellent collection of robotics algorithm implementations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
