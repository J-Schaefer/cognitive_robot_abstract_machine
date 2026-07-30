import logging

from giskardpy.motion_statechart.goals.templates import Parallel
from semantic_digital_twin.datastructures.definitions import GripperState

from griplink_interfaces.action import Grip, Release, Flexgrip, Flexrelease

from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    NavigateActionServerTask,
    ActionServerTask,
    WPGGripperActionServerTask,
)

from semantic_digital_twin.robots.daisy import DAiSy
from coraplex.datastructures.enums import ExecutionType, Arms
from coraplex.view_manager import ViewManager
from coraplex.robot_plans import (
    MoveMotion,
    MoveToolCenterPointMotion,
    LookingMotion,
    MoveGripperMotion,
)
from giskardpy.motion_statechart.graph_node import Task, MotionStatechartNode

from coraplex.robot_plans.motions.base import AlternativeMotion

logger = logging.getLogger(__name__)


class DAiSyGripMotion(MoveGripperMotion, AlternativeMotion[DAiSy]):
    """
    Uses the griplink action server to move the gripper of real DAiSy.
    """

    execution_type = ExecutionType.REAL

    def perform(self):
        logger.info(f"Performing action {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        if (
            self.motion == GripperState.FLEXOPEN
            or self.motion == GripperState.FLEXCLOSE
        ):
            raise ValueError(f"Gripper action {self.motion} not supported")

        tasks = []

        if self.gripper == Arms.LEFT:
            if self.motion == GripperState.OPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/release", message_type=Release
                    )
                )
            elif self.motion == GripperState.CLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/grip", message_type=Grip
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.RIGHT:
            if self.motion == GripperState.OPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/release", message_type=Release
                    )
                )
            elif self.motion == GripperState.CLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/grip", message_type=Grip
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.BOTH:
            if self.motion == GripperState.OPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/release", message_type=Release
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/release", message_type=Release
                    )
                )
            elif self.motion == GripperState.CLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/grip", message_type=Grip
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/grip", message_type=Grip
                    )
                )
        else:
            raise ValueError(f"Gripper {self.gripper} not supported")

        return Parallel(tasks)


class DAiSyFlexGripMotion(MoveGripperMotion, AlternativeMotion[DAiSy]):
    """
    Use flex grip and release motions for the WPG grippers.
    """

    execution_type = ExecutionType.REAL

    def perform(self):
        logger.info(f"Performing action {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        if self.motion == GripperState.OPEN or self.motion == GripperState.CLOSE:
            raise ValueError(f"Gripper action {self.motion} not supported")

        tasks = []

        if self.gripper == Arms.LEFT:
            if self.motion == GripperState.FLEXCLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexgrip", message_type=Flexgrip
                    )
                )
            elif self.motion == GripperState.FLEXOPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexrelease",
                        message_type=Flexrelease,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.RIGHT:
            if self.motion == GripperState.FLEXCLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexgrip", message_type=Flexgrip
                    )
                )
            elif self.motion == GripperState.FLEXOPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexrelease",
                        message_type=Flexrelease,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.BOTH:
            if self.motion == GripperState.FLEXCLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexgrip", message_type=Flexgrip
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexgrip", message_type=Flexgrip
                    )
                )
            elif self.motion == GripperState.FLEXOPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexrelease",
                        message_type=Flexrelease,
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexrelease",
                        message_type=Flexrelease,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        else:
            raise ValueError(f"Gripper {self.gripper} not supported")

        return Parallel(tasks)
