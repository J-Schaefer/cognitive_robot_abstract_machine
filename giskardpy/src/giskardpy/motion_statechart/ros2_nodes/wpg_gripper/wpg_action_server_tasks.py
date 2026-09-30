from __future__ import annotations

import logging
from dataclasses import dataclass

# Each task class commands its own pair of griplink actions.
from griplink_interfaces.action import Grip, Release
from griplink_interfaces.action import Flexgrip, Flexrelease

from typing import Generic

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    Action,
    ActionFeedback,
    ActionGoal,
    ActionResult,
    ActionServerTask,
)
from semantic_digital_twin.datastructures.robots.gripper_configurations import (
    WPGGripPreset,
)

logger = logging.getLogger(__name__)


# %% griplink goal defaults and status values

SUCCESS_STATUS = 0
"""
Gripper status the griplink server reports for a successfully finished action.
"""

DEFAULT_FLEXGRIP_POSITION_MM = 0
DEFAULT_FLEXGRIP_FORCE_N = 90
DEFAULT_FLEXGRIP_SPEED_MM_PER_S = 150
DEFAULT_FLEXGRIP_ACCELERATION_MM_PER_S2 = 600
DEFAULT_FLEXRELEASE_POSITION_MM = 120
DEFAULT_FLEXRELEASE_SPEED_MM_PER_S = 250
DEFAULT_FLEXRELEASE_ACCELERATION_MM_PER_S2 = 2000


# %% WPG griplink action server tasks


@dataclass(eq=False, repr=False)
class WPGActionServerTask(
    ActionServerTask[Action, ActionGoal, ActionResult, ActionFeedback],
    Generic[Action, ActionGoal, ActionResult, ActionFeedback],
):
    """
    Base class for tasks calling a WPG-300 griplink action server.

    Observes the gripper status the server reports in its result; subclasses build the
    goal for their pair of griplink actions.
    """

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        """
        Observes the gripper status the server reports once its result arrived.

        :param context: The motion statechart context the task runs in.
        :return:``TRUE`` when the server reported success, ``FALSE`` otherwise, and
            ``UNKNOWN`` while no result has arrived yet.
        """
        if self._result:
            gripper_status = self._result.result.status
            logger.info(f"Gripper status: {gripper_status}")
            return (
                ObservationStateValues.TRUE
                if gripper_status == SUCCESS_STATUS
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN


@dataclass(eq=False, repr=False)
class WPGGripActionServerTask(
    WPGActionServerTask[
        Grip | Release,
        Grip.Goal | Release.Goal,
        Grip.Result | Release.Result,
        Grip.Feedback | Release.Feedback,
    ]
):
    """
    Node for calling the griplink action server of a WPG gripper to execute a stored
    grip preset (``Grip``) or open the gripper (``Release``).
    """

    grip_preset: WPGGripPreset = WPGGripPreset.PRESET_0
    """
    Grip preset the server executes.
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Builds the ``Grip`` or ``Release`` goal selecting the configured preset.

        :param context: The motion statechart context the task runs in.
        :return: None; the goal is stored for :meth:`on_start` to send.
        :raises ValueError: If the task was built for neither ``Grip`` nor ``Release``.
        """
        if self.message_type is Grip:
            self._msg = Grip.Goal(
                port=0,
                index=self.grip_preset.value,
            )
        elif self.message_type is Release:
            self._msg = Release.Goal(
                port=0,
                index=self.grip_preset.value,
            )
        else:
            raise ValueError(f"Unknown message type: {self.message_type}")


@dataclass(eq=False, repr=False)
class WPGFlexActionServerTask(
    WPGActionServerTask[
        Flexgrip | Flexrelease,
        Flexgrip.Goal | Flexrelease.Goal,
        Flexgrip.Result | Flexrelease.Result,
        Flexgrip.Feedback | Flexrelease.Feedback,
    ]
):
    """
    Node for calling the griplink action server of a WPG gripper to flex grip to a
    commanded opening width (``Flexgrip``) or flex release from it (``Flexrelease``).
    """

    grip_position: int | None = None
    """
    Opening width of the gripper in mm [-5..120].

    Converted to µm when building the goal message.
    """

    grip_force: int | None = None
    """
    Force the gripper applies to the object in N [30..300].

    Converted to mN when building the Flexgrip goal message; ``Flexrelease`` has no
    force goal.
    """

    grip_speed: int | None = None
    """
    Motion speed of the gripper in mm/s [5..350].

    Converted to µm/s when building the goal message.
    """

    grip_acceleration: int | None = None
    """
    Motion acceleration of the gripper in mm/s² [100..4000].

    Converted to µm/s² when building the goal message.
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Builds the ``Flexgrip`` or ``Flexrelease`` goal, filling unset parameters with
        the defaults of the commanded action.

        :param context: The motion statechart context the task runs in.
        :return: None; the goal is stored for :meth:`on_start` to send.
        :raises ValueError: If the task was built for neither ``Flexgrip`` nor
            ``Flexrelease``.
        """
        if self.message_type is Flexgrip:
            position = (
                DEFAULT_FLEXGRIP_POSITION_MM
                if self.grip_position is None
                else self.grip_position
            )
            force = (
                DEFAULT_FLEXGRIP_FORCE_N if self.grip_force is None else self.grip_force
            )
            speed = (
                DEFAULT_FLEXGRIP_SPEED_MM_PER_S
                if self.grip_speed is None
                else self.grip_speed
            )
            acceleration = (
                DEFAULT_FLEXGRIP_ACCELERATION_MM_PER_S2
                if self.grip_acceleration is None
                else self.grip_acceleration
            )
            self._msg = Flexgrip.Goal(
                port=0,
                position=position * 1000,
                force=force * 1000,
                speed=speed * 1000,
                acceleration=acceleration * 1000,
            )
        elif self.message_type is Flexrelease:
            position = (
                DEFAULT_FLEXRELEASE_POSITION_MM
                if self.grip_position is None
                else self.grip_position
            )
            speed = (
                DEFAULT_FLEXRELEASE_SPEED_MM_PER_S
                if self.grip_speed is None
                else self.grip_speed
            )
            acceleration = (
                DEFAULT_FLEXRELEASE_ACCELERATION_MM_PER_S2
                if self.grip_acceleration is None
                else self.grip_acceleration
            )
            self._msg = Flexrelease.Goal(
                port=0,
                position=position * 1000,
                speed=speed * 1000,
                acceleration=acceleration * 1000,
            )
        else:
            raise ValueError(f"Unknown message type: {self.message_type}")
