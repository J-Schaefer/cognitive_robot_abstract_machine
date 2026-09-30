from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing_extensions import ClassVar, Generic, TypeVar

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.ros2_nodes.wpg_gripper.wpg_action_server_tasks import (
    WPGFlexActionServerTask,
    WPGGripActionServerTask,
)
from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.robots.gripper_configurations import (
    WPGGripperConfiguration,
)
from semantic_digital_twin.datastructures.robots.gripper_specification import (
    GripperSpecification,
    WPGFlexSpecification,
    WPGPresetSpecification,
)
from semantic_digital_twin.robots.daisy import (
    DAiSy,
    DAiSyLeftGripper,
    DAiSyRightGripper,
)

from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion

logger = logging.getLogger(__name__)

TSpecification = TypeVar("TSpecification", bound=GripperSpecification)


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


# %% DAiSy griplink motions


@dataclass
class DAiSyGripperMotion(MoveGripperMotion[TSpecification], Generic[TSpecification]):
    """
    Moves a WPG gripper of real DAiSy on its griplink action server, or commands a joint
    position goal for semi-real and simulated execution.

    Concrete motions are alternative motions for DAiSy and declare the griplink
    endpoints of the states they command.
    """

    execution_type: ClassVar[tuple[ExecutionType, ...]] = (
        ExecutionType.REAL,
        ExecutionType.SEMI_REAL,
        ExecutionType.SIMULATED,
    )

    _griplink_endpoints: ClassVar[dict[tuple[type, GripperState], WPGGripperEndpoint]]
    """
    Griplink endpoints the griplink server of each DAiSy WPG gripper and state listens
    on; concrete motions declare the table for the states they command.
    """

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

        return Parallel([self._action_server_task])

    @property
    def _griplink_endpoint(self) -> WPGGripperEndpoint:
        """
        :return: The endpoint the griplink server for this motion's gripper and state
            listens on.
        :raises ValueError: If the gripper or state has no endpoint in
            :attr:`_griplink_endpoints`.
        """
        state_type = self.specification.joint_state.state_type
        gripper_type = type(self.specification.end_effector)
        try:
            return self._griplink_endpoints[(gripper_type, state_type)]
        except KeyError:
            raise ValueError(
                f"Gripper action {state_type} not supported for {gripper_type.__name__}"
            )

    @property
    @abstractmethod
    def _action_server_task(self) -> MotionStatechartNode:
        """
        The griplink action server task this motion builds its chart from.
        """
        ...


# %% DAiSy grip motion


@dataclass
class DAiSyGripMotion(
    AlternativeMotion[DAiSy], DAiSyGripperMotion[WPGPresetSpecification]
):
    """
    Uses the griplink action server to grip or release with the WPG grippers of real
    DAiSy, or a joint position goal for semi-real execution.
    """

    _griplink_endpoints = {
        (DAiSyLeftGripper, GripperState.OPEN): WPGGripperEndpoint(
            action_topic="/left_gripper/release", message_type=Release
        ),
        (DAiSyLeftGripper, GripperState.CLOSE): WPGGripperEndpoint(
            action_topic="/left_gripper/grip", message_type=Grip
        ),
        (DAiSyRightGripper, GripperState.OPEN): WPGGripperEndpoint(
            action_topic="/right_gripper/release", message_type=Release
        ),
        (DAiSyRightGripper, GripperState.CLOSE): WPGGripperEndpoint(
            action_topic="/right_gripper/grip", message_type=Grip
        ),
    }

    @property
    def _action_server_task(self) -> WPGGripActionServerTask:
        """
        :return: The griplink task executing the preset of this motion's specification.
        """
        configuration: WPGGripperConfiguration = self.specification.configuration
        endpoint = self._griplink_endpoint
        return WPGGripActionServerTask(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_preset=configuration.grip_preset,
        )


# %% DAiSy flex grip motion


@dataclass
class DAiSyFlexGripMotion(
    AlternativeMotion[DAiSy], DAiSyGripperMotion[WPGFlexSpecification]
):
    """
    Uses flex grip and release motions for the WPG grippers of real DAiSy, or a joint
    position goal for semi-real execution.
    """

    _griplink_endpoints = {
        (DAiSyLeftGripper, GripperState.FLEXCLOSE): WPGGripperEndpoint(
            action_topic="/left_gripper/flexgrip", message_type=Flexgrip
        ),
        (DAiSyLeftGripper, GripperState.FLEXOPEN): WPGGripperEndpoint(
            action_topic="/left_gripper/flexrelease", message_type=Flexrelease
        ),
        (DAiSyRightGripper, GripperState.FLEXCLOSE): WPGGripperEndpoint(
            action_topic="/right_gripper/flexgrip", message_type=Flexgrip
        ),
        (DAiSyRightGripper, GripperState.FLEXOPEN): WPGGripperEndpoint(
            action_topic="/right_gripper/flexrelease", message_type=Flexrelease
        ),
    }

    @property
    def _action_server_task(self) -> WPGFlexActionServerTask:
        """
        :return: The griplink task executing the commanded opening width of this
            motion's specification.
        """
        configuration: WPGGripperConfiguration = self.specification.configuration
        endpoint = self._griplink_endpoint
        return WPGFlexActionServerTask(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_position=configuration.grip_position,
            grip_force=configuration.grip_force,
            grip_speed=configuration.grip_speed,
            grip_acceleration=configuration.grip_acceleration,
        )
