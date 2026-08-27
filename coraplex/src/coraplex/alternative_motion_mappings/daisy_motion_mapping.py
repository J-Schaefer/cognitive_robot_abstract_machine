from __future__ import annotations

import logging
from dataclasses import dataclass

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    WPGGripperActionServerTask,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.gripper_configurations import WPGGripperConfiguration
from semantic_digital_twin.robots.robot_parts import EndEffector

from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion
from coraplex.view_manager import ViewManager

logger = logging.getLogger(__name__)


# %% WPG endpoint resolution


@dataclass(frozen=True)
class WPGGripperEndpoint:
    """
    A griplink action server endpoint a single WPG gripper is reached on.
    """

    action_topic: str
    """
    ROS action topic the griplink server for one gripper listens on.
    """

    message_type: type
    """
    Griplink action message type this endpoint expects (``Grip``/``Release``/
    ``Flexgrip``/``Flexrelease``).
    """


def _resolve_wpg_endpoints(
    gripper: Arms,
    motion: GripperState,
    endpoint_table: dict,
) -> list[WPGGripperEndpoint]:
    """
    Resolve the griplink endpoint(s) for a gripper motion, expanding ``Arms.BOTH`` to
    one endpoint per physical gripper.

    :param gripper: The gripper side to move.
    :param motion: The gripper motion to perform.
    :param endpoint_table: Mapping from ``(Arms, GripperState)`` to ``(action_topic,
        message_type)``.
    :return: One endpoint per physical gripper involved.
    :raises ValueError: If the gripper or motion is not in the table.
    """
    if gripper == Arms.BOTH:
        sides = [Arms.LEFT, Arms.RIGHT]
    else:
        sides = [gripper]

    endpoints = []
    for side in sides:
        try:
            action_topic, message_type = endpoint_table[(side, motion)]
        except KeyError:
            raise ValueError(f"Gripper action {motion} not supported")
        endpoints.append(
            WPGGripperEndpoint(action_topic=action_topic, message_type=message_type)
        )
    return endpoints


_GRIP_ENDPOINTS: dict = {
    (Arms.LEFT, GripperState.OPEN): ("/left_gripper/release", Release),
    (Arms.LEFT, GripperState.CLOSE): ("/left_gripper/grip", Grip),
    (Arms.RIGHT, GripperState.OPEN): ("/right_gripper/release", Release),
    (Arms.RIGHT, GripperState.CLOSE): ("/right_gripper/grip", Grip),
}

_FLEX_ENDPOINTS: dict = {
    (Arms.LEFT, GripperState.FLEXCLOSE): ("/left_gripper/flexgrip", Flexgrip),
    (Arms.LEFT, GripperState.FLEXOPEN): ("/left_gripper/flexrelease", Flexrelease),
    (Arms.RIGHT, GripperState.FLEXCLOSE): ("/right_gripper/flexgrip", Flexgrip),
    (Arms.RIGHT, GripperState.FLEXOPEN): (
        "/right_gripper/flexrelease",
        Flexrelease,
    ),
}


def _resolved_wpg_configuration(motion: MoveGripperMotion) -> WPGGripperConfiguration:
    """
    Resolve the WPG gripper configuration for a motion, falling back to the
    configuration attached to the moved end effector.

    :param motion: The gripper motion to resolve the configuration for.
    :return: The resolved WPG gripper configuration.
    :raises ValueError: If no WPG gripper configuration is attached to the motion or its
        end effector.
    """
    configuration = motion.resolved_gripper_configuration()
    if not isinstance(configuration, WPGGripperConfiguration):
        raise ValueError(
            f"DAiSy gripper motion requires a {WPGGripperConfiguration.__name__}, "
            f"got {type(configuration).__name__ if configuration is not None else 'None'}"
        )
    return configuration


# %% DAiSy grip motion


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

        configuration = _resolved_wpg_configuration(self)
        endpoints = _resolve_wpg_endpoints(self.gripper, self.motion, _GRIP_ENDPOINTS)
        tasks = [
            WPGGripperActionServerTask(
                action_topic=endpoint.action_topic,
                message_type=endpoint.message_type,
                grip_preset=configuration.grip_preset,
            )
            for endpoint in endpoints
        ]
        return Parallel(tasks)


# %% DAiSy flex grip motion


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

    def perform(self):
        logger.info(f"Performing action {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        if self.motion == GripperState.OPEN or self.motion == GripperState.CLOSE:
            raise ValueError(f"Gripper action {self.motion} not supported")

        configuration = _resolved_wpg_configuration(self)

        if (
            GiskardExecutable.execution_type == ExecutionType.SEMI_REAL
            or GiskardExecutable.execution_type == ExecutionType.SIMULATED
        ):
            arm: EndEffector = ViewManager().get_end_effector_view(
                self.gripper, self.robot
            )
            position = (
                configuration.grip_position
                if configuration.grip_position is not None
                else 120
            )
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

        endpoints = _resolve_wpg_endpoints(self.gripper, self.motion, _FLEX_ENDPOINTS)
        tasks = [
            WPGGripperActionServerTask(
                action_topic=endpoint.action_topic,
                message_type=endpoint.message_type,
                grip_position=configuration.grip_position,
                grip_force=configuration.grip_force,
                grip_speed=configuration.grip_speed,
                grip_acceleration=configuration.grip_acceleration,
            )
            for endpoint in endpoints
        ]
        return Parallel(tasks)
