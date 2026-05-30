# === Phase 0: 依赖与常量 ===

import rclpy
from rclpy.node import Node
from smart_car_interfaces.msg import DecisionOutput, ControlOutput, VehicleFeedback, SystemCommand

TOPIC_CONTROL = "/smart_car/control/output"
TOPIC_DECISION = "/smart_car/decision/output"
TOPIC_VEHICLE_FB = "/smart_car/vehicle/feedback"
TOPIC_SYSTEM_CMD = "/smart_car/system/command"

# === Phase 1: 控制节点 ===

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.pub = self.create_publisher(ControlOutput, TOPIC_CONTROL, 10)
        self.pub_fb = self.create_publisher(VehicleFeedback, TOPIC_VEHICLE_FB, 10)
        self.sub_decision = self.create_subscription(DecisionOutput, TOPIC_DECISION, self._on_decision, 10)
        self.sub_cmd = self.create_subscription(SystemCommand, TOPIC_SYSTEM_CMD, self._on_cmd, 10)
        self._seq = 0

    def _on_decision(self, msg):
        out = ControlOutput()
        out.header.seq = self._seq
        self.pub.publish(out)
        self._seq += 1

    def _on_cmd(self, msg):
        pass

# === Phase 2: 入口 ===

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    main()
