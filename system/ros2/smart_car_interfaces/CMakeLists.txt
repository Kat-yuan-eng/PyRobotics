cmake_minimum_required(VERSION 3.8)
project(smart_car_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Header.msg"
  "msg/Vec2.msg"
  "msg/Vec3.msg"
  "msg/Pose2D.msg"
  "msg/Twist2D.msg"
  "msg/Obstacle.msg"
  "msg/RoadBoundary.msg"
  "msg/PerceptionOutput.msg"
  "msg/PathPoint.msg"
  "msg/TargetObject.msg"
  "msg/DecisionOutput.msg"
  "msg/SteeringCommand.msg"
  "msg/ThrottleCommand.msg"
  "msg/ControlOutput.msg"
  "msg/VehicleFeedback.msg"
  "msg/SystemCommand.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_package()
