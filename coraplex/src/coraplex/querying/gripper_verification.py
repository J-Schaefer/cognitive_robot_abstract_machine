from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.fragments.base import (
        VerbalizationFragment,
    )
    from krrood.entity_query_language.predicate import RenderedFields
    import rclpy

from krrood.entity_query_language.predicate import Predicate
from semantic_digital_twin.robots.robot_parts import EndEffector

try:
    import rclpy as _rclpy_module
except ImportError:
    _rclpy_module = None

try:
    from griplink_interfaces.srv import Devstate as DevstateSrv
except ImportError:
    DevstateSrv = None

logger = logging.getLogger(__name__)


class WPGGripperDeviceState(IntEnum):
    """
    Device state values reported by the WEISS WPG gripper via the devstate service.

    These correspond to the ``DeviceState`` enum in the GRIPLINK C++ driver.
    """

    NOT_CONNECTED = 0
    """
    No device connected.
    """

    NOT_INITIALIZED = 1
    """
    Device connected but not initialized.
    """

    DISABLED = 2
    """
    Device disabled.
    """

    RELEASED = 3
    """
    Gripper released (open).
    """

    NO_PART = 4
    """
    Gripper closed fully — no part was encountered.
    """

    HOLDING = 5
    """
    Gripper is holding a part.
    """

    OPERATING = 6
    """
    Gripper is operating (moving).
    """

    FAULT = 7
    """
    Device fault.
    """


def _arm_side_from_end_effector(end_effector: EndEffector) -> str:
    """
    Derive the arm side prefix (``"left"`` or ``"right"``) from the end effector type.

    :param end_effector: The end effector to inspect.
    :return:``"left"`` or ``"right"``.
    :raises ValueError: If the arm side cannot be determined.
    """
    class_name = type(end_effector).__name__.lower()
    if "left" in class_name:
        return "left"
    if "right" in class_name:
        return "right"
    raise ValueError(
        f"Cannot determine arm side from end effector type "
        f"{type(end_effector).__name__}"
    )


@dataclass(eq=False)
class IsGripperHoldingPart(Predicate):
    """
    Checks whether the WPG gripper is holding a part according to the hardware.

    Calls the ``devstate`` ROS2 service of the griplink driver and returns ``True`` if
    the reported device state is :attr:`~WPGGripperDeviceState.HOLDING`.
    """

    end_effector: EndEffector
    """
    The end effector whose gripper state should be queried.
    """

    ros_node: "rclpy.node.Node"
    """
    The ROS2 node used to make the service call.
    """

    service_name: Optional[str] = field(default=None)
    """
    Full name of the devstate service.

    If not provided, derived from the end effector type (e.g. ``/left_gripper/devstate``
    for a ``DAiSyLeftGripper``).
    """

    timeout: float = field(default=2.0)
    """
    Maximum time to wait for the service response in seconds.
    """

    def __post_init__(self):
        if self.service_name is None:
            side = _arm_side_from_end_effector(self.end_effector)
            self.service_name = f"/{side}_gripper/devstate"

    def _can_make_service_call(self) -> bool:
        if _rclpy_module is None:
            logger.warning("rclpy is not available; cannot query gripper state")
            return False
        if DevstateSrv is None:
            logger.warning(
                "griplink_interfaces is not available; cannot query gripper state"
            )
            return False
        if self.ros_node is None:
            logger.warning("ROS node is not available; cannot query gripper state")
            return False
        return True

    def _create_client(self):
        return self.ros_node.create_client(DevstateSrv, self.service_name)

    def __call__(self) -> bool:
        """
        Query the gripper device state and return whether the gripper is holding a part.

        :return:``True`` if the device state is ``HOLDING``, ``False`` if the service is
            unavailable or the state is not holding.
        """
        if not self._can_make_service_call():
            return False

        client = self._create_client()
        if not client.wait_for_service(timeout_sec=self.timeout):
            logger.warning("Devstate service %s not available", self.service_name)
            return False

        request = DevstateSrv.Request()
        request.port = 0
        future = client.call_async(request)
        _rclpy_module.spin_until_future_complete(
            self.ros_node, future, timeout_sec=self.timeout
        )

        if not future.done():
            logger.warning("Devstate service call timed out")
            return False

        response = future.result()
        return response.state == WPGGripperDeviceState.HOLDING

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
            clause,
            Copula,
            Noun,
        )

        return clause(
            Noun(fields["end_effector"]),
            Copula(),
            Noun("holding a part"),
        )


@dataclass(eq=False)
class IsGripperNotFullyClosed(Predicate):
    """
    Checks whether the gripper has not fully closed.

    Reads the current position of the gripper finger joint from the synced world state
    and compares it against a threshold. If the measured position is less than the
    threshold, something is likely blocking full closure.
    """

    end_effector: EndEffector
    """
    The end effector whose gripper state should be checked.
    """

    closed_position: float = field(default=0.038)
    """
    Joint position threshold in meters.

    Defaults to 0.038 m (DAiSy fully closed is 0.04 m, minus 2 mm margin). Positions
    below this value indicate the fingers did not fully close.
    """

    def _get_gripper_joint_name(self) -> str:
        """
        Derive the gripper finger joint name from the end effector type.

        :return: The joint name string.
        """
        side = _arm_side_from_end_effector(self.end_effector)
        return f"{side}_gripper_finger_joint"

    def __call__(self) -> bool:
        """
        Check whether the gripper is not fully closed.

        :return:``True`` if the measured joint position is less than
            :attr:`closed_position`.
        """
        joint_name = self._get_gripper_joint_name()
        connection = self.end_effector._world.get_connection_by_name(joint_name)
        current_position = self.end_effector._world.state[
            connection.raw_dof.id
        ].position
        return current_position < self.closed_position

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
            clause,
            Copula,
            Adjective,
            Noun,
        )

        return clause(
            Noun(fields["end_effector"]),
            Copula(),
            Adjective("not fully closed"),
        )
