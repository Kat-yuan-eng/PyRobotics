# === Phase 0: 依赖与常量 ===

import rclpy
from rclpy.node import Node
from smart_car_interfaces.msg import PerceptionOutput, SystemCommand

TOPIC_PERCEPTION = "/smart_car/perception/output"
TOPIC_SYSTEM_CMD = "/smart_car/system/command"

# === Phase 1: 感知节点 ===

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.pub = self.create_publisher(PerceptionOutput, TOPIC_PERCEPTION, 10)
        self.sub_cmd = self.create_subscription(SystemCommand, TOPIC_SYSTEM_CMD, self._on_cmd, 10)
        self.timer = self.create_timer(0.02, self._on_timer)
        self._seq = 0

    def _on_cmd(self, msg):
        pass

    def _on_timer(self):
        msg = PerceptionOutput()
        msg.header.seq = self._seq
        self.pub.publish(msg)
        self._seq += 1

# === Phase 2: 入口 ===

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    main()
