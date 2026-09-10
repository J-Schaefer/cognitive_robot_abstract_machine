from __future__ import annotations

import logging
from dataclasses import dataclass

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    WPGGripperActionServerTask,
)
from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.gripper_specification import (
    WPGFlexSpecification,
    WPGPresetSpecification,
)
from semantic_digital_twin.robots.daisy import (
    DAiSy,
    DAiSyLeftGripper,
    DAiSyRightGripper,
)
from semantic_digital_twin.robots.gripper_configurations import WPGGripperConfiguration
from semantic_digital_twin.robots.robot_parts import EndEffector

from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion

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


_GRIP_ENDPOINTS: dict = {
    (DAiSyLeftGripper, GripperState.OPEN): ("/left_gripper/release", Release),
    (DAiSyLeftGripper, GripperState.CLOSE): ("/left_gripper/grip", Grip),
    (DAiSyRightGripper, GripperState.OPEN): ("/right_gripper/release", Release),
    (DAiSyRightGripper, GripperState.CLOSE): ("/right_gripper/grip", Grip),
}

_FLEX_ENDPOINTS: dict = {
    (DAiSyLeftGripper, GripperState.FLEXCLOSE): (
        "/left_gripper/flexgrip",
        Flexgrip,
    ),
    (DAiSyLeftGripper, GripperState.FLEXOPEN): (
        "/left_gripper/flexrelease",
        Flexrelease,
    ),
    (DAiSyRightGripper, GripperState.FLEXCLOSE): (
        "/right_gripper/flexgrip",
        Flexgrip,
    ),
    (DAiSyRightGripper, GripperState.FLEXOPEN): (
        "/right_gripper/flexrelease",
        Flexrelease,
    ),
}


def _resolve_wpg_endpoint(
    end_effector: EndEffector,
    state_type: GripperState,
    endpoint_table: dict,
) -> WPGGripperEndpoint:
    """
    Resolve the griplink endpoint for one gripper motion.

    :param end_effector: The WPG gripper to move.
    :param state_type: The gripper state to command.
    :param endpoint_table: Mapping from ``(gripper_class, GripperState)`` to
        ``(action_topic, message_type)``.
    :return: The endpoint for that gripper and state.
    :raises ValueError: If the gripper or state is not in the table.
    """
    try:
        action_topic, message_type = endpoint_table[(type(end_effector), state_type)]
    except KeyError:
        raise ValueError(
            f"Gripper action {state_type} not supported for {type(end_effector).__name__}"
        )
    return WPGGripperEndpoint(action_topic=action_topic, message_type=message_type)


# %% DAiSy grip motion


@dataclass
class DAiSyGripMotion(
    AlternativeMotion[DAiSy], MoveGripperMotion[WPGPresetSpecification]
):
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
            GiskardExecutable.execution_type == ExecutionType.SEMI_REAL
            or GiskardExecutable.execution_type == ExecutionType.SIMULATED
        ):
            return super()._motion_chart

        configuration: WPGGripperConfiguration = self.specification.configuration
        state_type = self.specification.joint_state.state_type
        endpoint = _resolve_wpg_endpoint(
            self.specification.end_effector, state_type, _GRIP_ENDPOINTS
        )
        return Parallel(
            [
                WPGGripperActionServerTask(
                    action_topic=endpoint.action_topic,
                    message_type=endpoint.message_type,
                    grip_preset=configuration.grip_preset,
                )
            ]
        )


# %% DAiSy flex grip motion


@dataclass
class DAiSyFlexGripMotion(
    AlternativeMotion[DAiSy], MoveGripperMotion[WPGFlexSpecification]
):
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
        if (
            GiskardExecutable.execution_type == ExecutionType.SEMI_REAL
            or GiskardExecutable.execution_type == ExecutionType.SIMULATED
        ):
            return super()._motion_chart

        configuration: WPGGripperConfiguration = self.specification.configuration
        state_type = self.specification.joint_state.state_type
        endpoint = _resolve_wpg_endpoint(
            self.specification.end_effector, state_type, _FLEX_ENDPOINTS
        )
        return Parallel(
            [
                WPGGripperActionServerTask(
                    action_topic=endpoint.action_topic,
                    message_type=endpoint.message_type,
                    grip_position=configuration.grip_position,
                    grip_force=configuration.grip_force,
                    grip_speed=configuration.grip_speed,
                    grip_acceleration=configuration.grip_acceleration,
                )
            ]
        )
