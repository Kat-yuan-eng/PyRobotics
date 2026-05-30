# === Phase 0: 依赖与常量 ===

import rclpy
from rclpy.node import Node
from smart_car_interfaces.msg import PerceptionOutput, DecisionOutput, SystemCommand
from message_filters import Subscriber, ApproximateTimeSynchronizer

TOPIC_PERCEPTION = "/smart_car/perception/output"
TOPIC_DECISION = "/smart_car/decision/output"
TOPIC_SYSTEM_CMD = "/smart_car/system/command"

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        self.pub = self.create_publisher(DecisionOutput, TOPIC_DECISION, 10)
        sub_percep = Subscriber(self, PerceptionOutput, TOPIC_PERCEPTION)
        sub_cmd = Subscriber(self, SystemCommand, TOPIC_SYSTEM_CMD)
        self.sync = ApproximateTimeSynchronizer(
            [sub_percep, sub_cmd], queue_size=10, slop=0.05)
        self.sync.registerCallback(self._on_synced)
        self._seq = 0

    def _on_synced(self, percep_msg, cmd_msg):
        out = DecisionOutput()
        out.header.seq = self._seq
        self.pub.publish(out)
        self._seq += 1

# === Phase 2: 入口 ===

def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    main()
