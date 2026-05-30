# === Phase 0: Launch配置 ===

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    perception = Node(
        package='smart_car_control_system',
        executable='perception_node',
        name='perception_node',
        output='screen',
    )
    planning = Node(
        package='smart_car_control_system',
        executable='planning_node',
        name='planning_node',
        output='screen',
    )
    control = Node(
        package='smart_car_control_system',
        executable='control_node',
        name='control_node',
        output='screen',
    )
    return LaunchDescription([perception, planning, control])
