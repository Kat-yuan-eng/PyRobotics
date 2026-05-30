# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-29

### Added

- **Perception Layer**: Lane pixel detector, obstacle detector (RANSAC + DBSCAN), obstacle tracker (Hungarian matching), sign recognizer (HOG + template), sensor fusion (pinhole model back-projection)
- **Decision Layer**: Task scheduler (FSM: PATROL→AVOID→PARK), path smoother (curvature-constrained), obstacle avoidance (lateral offset), multi-agent coordination
- **Path Planning**: A* grid planner, RRT planner (with path smoothing), DWA planner (with global path alignment cost)
- **Path Tracking**: Stanley (front-axle feedback), Pure Pursuit, Fuzzy (curvature-adaptive), MPC (terminal cost), DQN (EMA smoothing + safety filter), controller selector (Bumpless Transfer)
- **Localization**: EKF (Joseph-form update), Particle Filter (systematic resampling), Covariance Intersection Fusion
- **SLAM**: FastSLAM 2.0 (adaptive resampling threshold, Joseph-form landmark update), ICP matching (init pose guess + Huber kernel), SLAM pipeline
- **System**: Vehicle simulator (bicycle model), real-time pipeline (dual-thread), embedded C (PID, Stanley, EKF, vehicle control with Q16.16 fixed-point), ROS2 nodes (ApproximateTimeSynchronizer), Protobuf message definitions
- **Safety**: Throttle/brake mutual exclusion, EKF covariance explosion detection, NaN guards, obstacle coordinate validation, ID collision prevention
- **Testing**: 27 unit tests covering all modules
- **Documentation**: Architecture, API reference, quick start guide (Chinese)
