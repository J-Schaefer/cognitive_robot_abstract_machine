from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from giskardpy.motion_statechart.goals.templates import Parallel
from semantic_digital_twin.datastructures.definitions import GripperState

from griplink_interfaces.action import Grip, Release, Flexgrip, Flexrelease

from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    NavigateActionServerTask,
    ActionServerTask,
    WPGGripperActionServerTask,
)

from semantic_digital_twin.robots.daisy import DAiSy
from coraplex.datastructures.enums import ExecutionType, Arms, WPGGripPreset
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


@dataclass
class DAiSyGripMotion(MoveGripperMotion, AlternativeMotion[DAiSy]):
    """
    Uses the griplink action server to move the gripper of real DAiSy.
    """

    execution_type = ExecutionType.REAL

    grip_preset: WPGGripPreset = WPGGripPreset.PRESET_0
    """
    Grip preset index passed to the Grip/Release action.
    """

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

        task_kwargs = dict(
            grip_preset=self.grip_preset,
        )

        tasks = []

        if self.gripper == Arms.LEFT:
            if self.motion == GripperState.OPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/release",
                        message_type=Release,
                        **task_kwargs,
                    )
                )
            elif self.motion == GripperState.CLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/grip",
                        message_type=Grip,
                        **task_kwargs,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.RIGHT:
            if self.motion == GripperState.OPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/release",
                        message_type=Release,
                        **task_kwargs,
                    )
                )
            elif self.motion == GripperState.CLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/grip",
                        message_type=Grip,
                        **task_kwargs,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.BOTH:
            if self.motion == GripperState.OPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/release",
                        message_type=Release,
                        **task_kwargs,
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/release",
                        message_type=Release,
                        **task_kwargs,
                    )
                )
            elif self.motion == GripperState.CLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/grip",
                        message_type=Grip,
                        **task_kwargs,
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/grip",
                        message_type=Grip,
                        **task_kwargs,
                    )
                )
        else:
            raise ValueError(f"Gripper {self.gripper} not supported")

        return Parallel(tasks)


@dataclass
class DAiSyFlexGripMotion(MoveGripperMotion, AlternativeMotion[DAiSy]):
    """
    Use flex grip and release motions for the WPG grippers.
    """

    execution_type = ExecutionType.REAL

    grip_position: Optional[int] = None
    """
    Opening width of the gripper [-5..120 mm].
    """

    grip_force: Optional[int] = None
    """
    Force the gripper applies to the object [30..300 N].
    """

    grip_speed: Optional[int] = None
    """
    Motion speed of the gripper [5..350 mm/s].
    """

    grip_acceleration: Optional[int] = None
    """
    Motion acceleration of the gripper [100..4000 mm/s^2].
    """

    def perform(self):
        logger.info(f"Performing action {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        if self.motion == GripperState.OPEN or self.motion == GripperState.CLOSE:
            raise ValueError(f"Gripper action {self.motion} not supported")

        task_kwargs = dict(
            grip_position=self.grip_position,
            grip_force=self.grip_force,
            grip_speed=self.grip_speed,
            grip_acceleration=self.grip_acceleration,
        )

        tasks = []

        if self.gripper == Arms.LEFT:
            if self.motion == GripperState.FLEXCLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexgrip",
                        message_type=Flexgrip,
                        **task_kwargs,
                    )
                )
            elif self.motion == GripperState.FLEXOPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexrelease",
                        message_type=Flexrelease,
                        **task_kwargs,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.RIGHT:
            if self.motion == GripperState.FLEXCLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexgrip",
                        message_type=Flexgrip,
                        **task_kwargs,
                    )
                )
            elif self.motion == GripperState.FLEXOPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexrelease",
                        message_type=Flexrelease,
                        **task_kwargs,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        elif self.gripper == Arms.BOTH:
            if self.motion == GripperState.FLEXCLOSE:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexgrip",
                        message_type=Flexgrip,
                        **task_kwargs,
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexgrip",
                        message_type=Flexgrip,
                        **task_kwargs,
                    )
                )
            elif self.motion == GripperState.FLEXOPEN:
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/left_gripper/flexrelease",
                        message_type=Flexrelease,
                        **task_kwargs,
                    )
                )
                tasks.append(
                    WPGGripperActionServerTask(
                        action_topic="/right_gripper/flexrelease",
                        message_type=Flexrelease,
                        **task_kwargs,
                    )
                )
            else:
                raise ValueError(f"Gripper action {self.motion} not supported")
        else:
            raise ValueError(f"Gripper {self.gripper} not supported")

        return Parallel(tasks)
