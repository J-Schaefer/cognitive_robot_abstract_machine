from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from griplink_interfaces.action import Grip, Release, Flexgrip, Flexrelease

from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    NavigateActionServerTask,
    ActionServerTask,
    WPGGripperActionServerTask,
)

from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import EndEffector
from coraplex.datastructures.enums import ExecutionType, Arms, WPGGripPreset
from coraplex.plans.executables import GiskardExecutable
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
    Uses the griplink action server to move the gripper of real DAiSy, or a joint
    position goal for semi-real execution.
    """

    execution_type = (
        ExecutionType.REAL,
        ExecutionType.SEMI_REAL,
        ExecutionType.SIMULATED,
    )

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

        if (
            GiskardExecutable.execution_type == ExecutionType.SEMI_REAL
            or GiskardExecutable.execution_type == ExecutionType.SIMULATED
        ):
            arm: EndEffector = ViewManager().get_end_effector_view(
                self.gripper, self.robot
            )
            return JointPositionList(
                goal_state=arm.get_joint_state_by_type(self.motion),
                name=(
                    "OpenGripper"
                    if self.motion == GripperState.OPEN
                    else "CloseGripper"
                ),
            )

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
    Use flex grip and release motions for the WPG grippers, or a joint position goal for
    semi-real execution.
    """

    execution_type = (
        ExecutionType.REAL,
        ExecutionType.SEMI_REAL,
        ExecutionType.SIMULATED,
    )

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

        if (
            GiskardExecutable.execution_type == ExecutionType.SEMI_REAL
            or GiskardExecutable.execution_type == ExecutionType.SIMULATED
        ):
            arm: EndEffector = ViewManager().get_end_effector_view(
                self.gripper, self.robot
            )
            position = self.grip_position if self.grip_position is not None else 120
            open_state = arm.get_joint_state_by_type(GripperState.OPEN)
            fraction = (120 - position) / 120
            target_values = []
            for connection in open_state.connections:
                lower = connection.dof.limits.lower.position or 0.0
                upper = connection.dof.limits.upper.position or 0.0
                sdt_position = lower + fraction * (upper - lower)
                target_values.append(sdt_position)
            joint_state = JointState(
                connections=open_state.connections,
                target_values=target_values,
                state_type=self.motion,
                name=PrefixedName("flexgrip", prefix=arm.name.name),
            )
            return JointPositionList(
                goal_state=joint_state,
                name=(
                    "FlexOpenGripper"
                    if self.motion == GripperState.FLEXOPEN
                    else "FlexCloseGripper"
                ),
            )

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
