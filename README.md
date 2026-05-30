<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=autonomous%20vehicle%20software%20architecture%20diagram%20four%20layer%20perception%20decision%20control%20system%20clean%20technical%20blue%20white&image_size=landscape_16_9" align="right" width="300" alt="PyRobotics header pic"/>

# PyRobotics
![GitHub_Action_CI](https://github.com/user/PyRobotics/workflows/CI/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](pyproject.toml)

Python codes for intelligent vehicle autonomous driving with a four-layer architecture: **Perception → Decision → Control → System**.


# Table of Contents
   * [What is this?](#what-is-this)
   * [Requirements](#requirements)
   * [How to use](#how-to-use)
   * [Perception](#perception)
      * [Lane pixel detection](#lane-pixel-detection)
      * [Obstacle detection](#obstacle-detection)
      * [Obstacle tracking](#obstacle-tracking)
      * [Sign recognition](#sign-recognition)
      * [Sensor fusion](#sensor-fusion)
   * [Decision](#decision)
      * [Task scheduling](#task-scheduling)
      * [Path smoothing](#path-smoothing)
      * [Obstacle avoidance](#obstacle-avoidance)
      * [Multi-agent coordination](#multi-agent-coordination)
   * [Path Planning](#path-planning)
      * [A* algorithm](#a-algorithm)
      * [RRT planner](#rrt-planner)
      * [Dynamic Window Approach](#dynamic-window-approach)
   * [Path Tracking](#path-tracking)
      * [Stanley control](#stanley-control)
      * [Pure Pursuit control](#pure-pursuit-control)
      * [Fuzzy control](#fuzzy-control)
      * [Model predictive control](#model-predictive-control)
      * [DQN control](#dqn-control)
      * [Controller selection](#controller-selection)
   * [Localization](#localization)
      * [Extended Kalman Filter localization](#extended-kalman-filter-localization)
      * [Particle filter localization](#particle-filter-localization)
      * [Covariance Intersection fusion](#covariance-intersection-fusion)
   * [SLAM](#slam)
      * [FastSLAM 2.0](#fastslam-20)
      * [Iterative Closest Point (ICP) Matching](#iterative-closest-point-icp-matching)
      * [SLAM pipeline](#slam-pipeline)
   * [System](#system)
      * [Vehicle simulation](#vehicle-simulation)
      * [Real-time pipeline](#real-time-pipeline)
      * [Embedded C implementation](#embedded-c-implementation)
      * [ROS2 integration](#ros2-integration)
   * [License](#license)
   * [Contribution](#contribution)
   * [Authors](#authors)

# What is PyRobotics?

PyRobotics is a Python code collection for intelligent vehicle autonomous driving, implementing a complete four-layer software stack.

Features:

1. Complete autonomous driving pipeline: Perception → Decision → Control → System.

2. All modules decoupled via Protobuf messages with strict linear data flow.

3. Multiple algorithm choices per layer (5 tracking controllers, 3 planners, 3 localizers).

4. Safety-critical design: Joseph-form covariance, throttle/brake mutual exclusion, NaN guards.

5. Minimum dependency (NumPy, SciPy, Matplotlib, OpenCV, scikit-learn, protobuf).

Inspired by [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) — a Python code collection of robotics algorithms.


# Requirements

For running each sample code:

- [Python 3.9+](https://www.python.org/)

- [NumPy](https://numpy.org/)

- [SciPy](https://scipy.org/)

- [Matplotlib](https://matplotlib.org/)

- [OpenCV](https://opencv.org/)

- [scikit-learn](https://scikit-learn.org/)

- [protobuf](https://protobuf.dev/)

For development:

- [pytest](https://pytest.org/) (for unit tests)

- [grpcio-tools](https://grpc.io/) (for Protobuf code generation)

- [ruff](https://docs.astral.sh/ruff/) (for code style check)

- [mypy](https://mypy-lang.org/) (for type check)


# How to use

1. Clone this repo.

   ```terminal
   git clone https://github.com/user/PyRobotics.git
   ```


2. Install the required libraries.

   ```terminal
   pip install -r requirements/requirements.txt
   ```

3. (Optional) Regenerate Protobuf code if you modify `.proto` files.

   ```terminal
   python -m grpc_tools.protoc -I=proto --python_out=generated proto/*.proto
   ```


4. Execute python script in each directory.

5. Add star to this repo if you like it :smiley:.


# Perception

## Lane pixel detection

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=lane%20detection%20on%20road%20image%20green%20overlay%20on%20white%20lane%20markers%20top%20view%20clean&image_size=landscape_16_9" width="640" alt="Lane detection pic">

This is a lane pixel detection module using HLS color space conversion and scan-line peak detection with jump filter.

It extracts lane center line and boundary points from camera images.

## Obstacle detection

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=3D%20point%20cloud%20obstacle%20detection%20with%20colored%20clusters%20RANSAC%20ground%20removal%20DBSCAN%20clustering%20clean%20technical&image_size=landscape_16_9" width="640" alt="Obstacle detection pic">

This is a 3D obstacle detection module using RANSAC ground plane removal, DBSCAN clustering, and OBB (Oriented Bounding Box) fitting.

It outputs obstacle center, size, and orientation for downstream decision-making.

## Obstacle tracking

This is a multi-object tracking module using Hungarian algorithm for data association and Kalman filter for velocity estimation.

It maintains track IDs across frames and handles track creation/deletion.

## Sign recognition

This is a traffic sign recognition module using HOG (Histogram of Oriented Gradients) descriptors and template matching.

It detects and classifies speed limit and stop signs from camera images.

## Sensor fusion

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=multi%20sensor%20fusion%20diagram%20camera%20lidar%20pinhole%20model%20back%20projection%20vehicle%20coordinates%20technical&image_size=landscape_16_9" width="640" alt="Sensor fusion pic">

This is a multi-sensor fusion module using pinhole camera model back-projection.

It transforms detections from camera pixel coordinates to vehicle coordinates, combining camera and LiDAR information into a unified PerceptionOutput message.

# Decision

## Task scheduling

This is a finite state machine (FSM) based task scheduler with three states: PATROL → AVOID → PARK.

It generates DecisionOutput messages containing target path, speed, and behavior for the control layer.

## Path smoothing

This is a curvature-constrained path smoothing module using iterative shortening with deviation bounds.

It ensures the smoothed path stays within `max_deviation` of the original while producing continuous curvature profiles for downstream controllers.

## Obstacle avoidance

This is a lateral offset obstacle avoidance module that generates parallel shifted paths around detected obstacles.

It computes safe lateral offset based on obstacle size and safety margin.

## Multi-agent coordination

This is a multi-agent coordination module with ID-offset assignment and velocity validation.

It prevents ID collisions between agents and validates speed/timestamp consistency.

# Path Planning

## A* algorithm

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=A%20star%20path%20planning%20on%20grid%20map%20cyan%20searched%20nodes%20red%20path%20black%20obstacles%20green%20start%20goal&image_size=landscape_16_9" width="640" alt="A* pic">

This is a 2D grid based shortest path planning with A* algorithm.

In the animation, cyan points are searched nodes, red line is the planned path.

Reference

- [A* search algorithm - Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm)

## RRT planner

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=RRT%20rapidly%20exploring%20random%20tree%20path%20planning%20green%20tree%20red%20path%20black%20obstacles%20blue%20smoothed&image_size=landscape_16_9" width="640" alt="RRT pic">

This is a sampling-based path planning with RRT (Rapidly-Exploring Random Trees) and integrated path smoothing.

Black circles are obstacles, green line is the search tree, red line is the smoothed path.

Reference

- [RRT - Wikipedia](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree)

## Dynamic Window Approach

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=dynamic%20window%20approach%20DWA%20local%20planner%20trajectory%20rollout%20obstacle%20avoidance%20robot%20navigation&image_size=landscape_16_9" width="640" alt="DWA pic">

This is a 2D navigation with Dynamic Window Approach including global path alignment cost.

It evaluates candidate trajectories considering heading, clearance, speed, and global path consistency.

Reference

- [The Dynamic Window Approach to Collision Avoidance](https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf)


# Path Tracking

## Stanley control

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Stanley%20controller%20path%20tracking%20front%20axle%20lateral%20error%20feedback%20vehicle%20following%20reference%20path&image_size=landscape_16_9" width="640" alt="Stanley pic">

Path tracking simulation with Stanley steering control (front-axle feedback) and PID speed control.

The blue line is the reference path, the red line is the actual trajectory.

Reference

- [Stanley: The robot that won the DARPA grand challenge](http://robots.stanford.edu/papers/thrun.stanley05.pdf)

- [Automatic Steering Methods for Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)


## Pure Pursuit control

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=pure%20pursuit%20controller%20lookahead%20circle%20arc%20path%20tracking%20vehicle%20bicycle%20model&image_size=landscape_16_9" width="640" alt="Pure Pursuit pic">

Path tracking simulation with Pure Pursuit steering control and PID speed control.

The blue line is the reference path, the green circle is the lookahead distance, the red line is the actual trajectory.

Reference

- [Implementation of the Pure Pursuit Path Tracking Algorithm](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)


## Fuzzy control

This is a curvature-adaptive fuzzy controller with complete membership function overlap.

It uses curvature (κ) and speed deviation (Δv) as inputs, with 3×5 rule base for steering and throttle control.

The membership functions are designed to eliminate boundary dead zones.

## Model predictive control

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=model%20predictive%20control%20MPC%20path%20tracking%20predicted%20trajectory%20horizon%20vehicle%20reference%20path&image_size=landscape_16_9" width="640" alt="MPC pic">

Path tracking simulation with iterative linear Model Predictive Control including terminal cost.

It optimizes steering and speed over a prediction horizon with explicit constraint handling.

Reference

- [Real-time Model Predictive Control (MPC)](http://grauonline.de/wordpress/?page_id=3244)


## DQN control

This is a Deep Q-Network (DQN) based controller with EMA action smoothing and safety filter.

The Q-network outputs discrete steering actions, smoothed by exponential moving average (α=0.3).

A safety filter overrides unsafe actions based on lateral error and heading error thresholds.

## Controller selection

This is a speed/curvature-based controller selector with Bumpless Transfer.

It switches between Stanley (high speed), Pure Pursuit (medium), and Fuzzy (low speed/curvature) based on current driving conditions.

Bumpless Transfer ensures smooth steering transitions by passing `prev_steer` between controllers.


# Localization

## Extended Kalman Filter localization

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=extended%20kalman%20filter%20localization%20blue%20true%20trajectory%20red%20estimated%20black%20dead%20reckoning%20landmarks&image_size=landscape_16_9" width="640" alt="EKF pic">

This is a sensor fusion localization with Extended Kalman Filter (EKF).

The blue line is true trajectory, the black line is dead reckoning trajectory,

and the red line is an estimated trajectory with EKF.

It uses Joseph-form covariance update to guarantee positive definiteness.

Covariance explosion detection resets P when max(diag) > 100.

Reference

- [PROBABILISTIC ROBOTICS](http://www.probabilistic-robotics.org/)


## Particle filter localization

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=particle%20filter%20localization%20red%20particles%20blue%20true%20trajectory%20black%20dead%20reckoning%20landmarks&image_size=landscape_16_9" width="640" alt="PF pic">

This is a sensor fusion localization with Particle Filter (PF).

The blue line is true trajectory, the black line is dead reckoning trajectory,

and the red line is an estimated trajectory with PF.

It uses systematic resampling with NaN-safe weight normalization.

Reference

- [PROBABILISTIC ROBOTICS](http://www.probabilistic-robotics.org/)


## Covariance Intersection fusion

This is a multi-estimator fusion module using Covariance Intersection (CI).

It combines EKF and PF estimates without requiring cross-correlation knowledge.

Divergence detection falls back to the more consistent estimator when one diverges.


# SLAM

Simultaneous Localization and Mapping (SLAM) examples

## FastSLAM 2.0

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=FastSLAM%202.0%20SLAM%20blue%20ground%20truth%20red%20estimated%20trajectory%20black%20landmarks%20blue%20crosses%20particles&image_size=landscape_16_9" width="640" alt="FastSLAM pic">

This is a feature based SLAM example using FastSLAM 2.0 with Joseph-form landmark update.

The blue line is ground truth, the black line is dead reckoning, the red line is the estimated trajectory with FastSLAM.

The red points are particles of FastSLAM.

Black points are landmarks, blue crosses are estimated landmark positions by FastSLAM.

It uses adaptive resampling threshold based on EMA of effective sample size.

Reference

- [PROBABILISTIC ROBOTICS](http://www.probabilistic-robotics.org/)

- [SLAM simulations by Tim Bailey](http://www-personal.acfr.usyd.edu.au/tbailey/software/slam_simulations.htm)


## Iterative Closest Point (ICP) Matching

<img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=ICP%20iterative%20closest%20point%20matching%20two%20point%20clouds%20alignment%20before%20after%20convergence&image_size=landscape_16_9" width="640" alt="ICP pic">

This is a 2D ICP matching example with singular value decomposition.

It supports initial pose guess and Huber robust kernel for outlier rejection.

It can calculate a rotation matrix and a translation vector between point clouds.

Reference

- [Introduction to Mobile Robotics: Iterative Closest Point Algorithm](https://cs.gmu.edu/~kosecka/cs685/cs685-icp.pdf)


## SLAM pipeline

This is an integrated SLAM pipeline combining EKF localization, FastSLAM 2.0, and ICP refinement.

It runs EKF for real-time localization and FastSLAM for map building, with periodic ICP loop closure.


# System

## Vehicle simulation

This is a bicycle kinematic model vehicle simulator.

It supports front-wheel steering with wheelbase compensation and provides state feedback for closed-loop control.

## Real-time pipeline

This is a dual-thread real-time pipeline with perception thread (30ms period) and control thread (50Hz).

It uses thread-safe LatestResult containers for inter-thread communication.

## Embedded C implementation

This is a set of embedded C implementations with Q16.16 fixed-point arithmetic:

- PID controller with anti-windup and jerk limiting
- Stanley steering with front-axle feedback
- EKF localization with Joseph-form update
- Vehicle control with inner/outer loop PID

All fixed-point operations include overflow protection (FP_CLIP).

## ROS2 integration

This is a ROS2 integration package with three nodes:

- Perception node: processes camera/LiDAR data
- Planning node: uses ApproximateTimeSynchronizer for multi-topic alignment
- Control node: outputs vehicle commands

It includes custom message definitions in `smart_car_interfaces`.


# License

MIT


# Contribution

Any contribution is welcome!!

Please check [CONTRIBUTING.md](CONTRIBUTING.md) for details.


# Authors

- [PyRobotics Contributors](https://github.com/user/PyRobotics/graphs/contributors)

- Inspired by [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) by Atsushi Sakai
