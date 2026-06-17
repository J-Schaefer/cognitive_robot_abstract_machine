from semantic_digital_twin.datastructures.definitions import GripperState

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    NavigateActionServerTask,
    ActionServerTask, WPGGripperActionServerTask,
)
from semantic_digital_twin.robots.daisy import DAiSy
from coraplex.datastructures.enums import ExecutionType, Arms
from coraplex.view_manager import ViewManager
from coraplex.robot_plans import MoveMotion, MoveToolCenterPointMotion, LookingMotion, MoveGripperMotion

from coraplex.robot_plans.motions.base import AlternativeMotion


class DAISYGripMotion(MoveGripperMotion, AlternativeMotion[DAiSy]):
    """
    Uses the griplink action server to move the gripper of real DAiSy
    """

    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> ActionServerTask:

        if self.gripper == Arms.LEFT:
            if self.motion == GripperState.OPEN:
                self.action_topic = "/left_gripper/release"
                self.message_type = NavigateToPose
            elif self.motion == GripperState.CLOSE:
                self.action_topic = "/left_gripper/grip"
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.RIGHT:
            if self.motion == GripperState.OPEN:
                self.action_topic = "/right_gripper/release"
            elif self.motion == GripperState.CLOSE:
                self.action_topic = "/right_gripper/grip"
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        else:
            raise ValueError(f"Gripper {self.gripper} not supported")

        if self.motion == GripperState.FLEXOPEN or self.motion == GripperState.FLEXCLOSE:
            return WPGGripperActionServerTask(action_topic=self.action_topic, )

    class DaisyFlexGripMotion(MoveGripperMotion, AlternativeMotion[DAiSy]):
        """
        Use flex grip and release motions for the WPG grippers.
        """

        def perform(self):
            return

        def _motion_chart(self) -> ActionServerTask:
            if self.gripper == Arms.LEFT:
                if self.motion == GripperState.FLEXCLOSE:
                    self.action_topic = "/left_gripper/flexgrip"
                elif self.motion == GripperState.FLEXOPEN:
                    self.action_topic = "/left_gripper/flexrelease"
                else:
                    raise ValueError(f"Gripper action {self.motion} not supported")
            elif self.gripper == Arms.RIGHT:
                if self.motion == GripperState.FLEXCLOSE:
                    self.action_topic = "/right_gripper/flexgrip"
                elif self.motion == GripperState.FLEXOPEN:
                    self.action_topic = "/right_gripper/flexrelease"
                else:
                    raise ValueError(f"Gripper action {self.motion} not supported")
            else:
                raise ValueError(f"Gripper {self.gripper} not supported")

            if self.motion == GripperState.FLEXOPEN or self.motion == GripperState.FLEXCLOSE:
                return WPGGripperActionServerTask(action_topic=self.action_topic, )
