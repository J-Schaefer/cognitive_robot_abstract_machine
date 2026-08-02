from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Type, Optional, Callable, TypeVar

from rclpy.action import ActionServer
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from coraplex.querying.gripper_verification import WPGGripperDeviceState
from giskardpy.middleware.ros2 import rospy

logger = logging.getLogger(__name__)

try:
    from griplink_interfaces.action import Grip, Release, Flexgrip, Flexrelease

    GIPLINK_AVAILABLE = True
except ImportError:
    GIPLINK_AVAILABLE = False
    logger.warning("griplink_interfaces not available, WPG fake servers will not work")

try:
    from griplink_interfaces.srv import Devstate

    GIPLINK_SRV_AVAILABLE = True
except ImportError:
    GIPLINK_SRV_AVAILABLE = False


@dataclass
class GripperConnector:
    """
    Configuration for gripper joint connector in the fake action server.
    """

    joint_names: list[str]
    """
    List of joint names that this gripper controls.
    """

    open_positions: float | list[float]
    """
    Joint position(s) for the open gripper state.

    Can be a single value for all joints or a list matching the length of joint_names.
    """

    closed_positions: float | list[float]
    """
    Joint position(s) for the closed gripper state.

    Can be a single value for all joints or a list matching the length of joint_names.
    """

    def __post_init__(self):
        if isinstance(self.open_positions, (int, float)):
            self.open_positions = [self.open_positions] * len(self.joint_names)
        if isinstance(self.closed_positions, (int, float)):
            self.closed_positions = [self.closed_positions] * len(self.joint_names)

        if len(self.open_positions) != len(self.joint_names):
            raise ValueError(
                f"Open positions length ({len(self.open_positions)}) must match "
                f"joint names length ({len(self.joint_names)})"
            )
        if len(self.closed_positions) != len(self.joint_names):
            raise ValueError(
                f"Closed positions length ({len(self.closed_positions)}) must match "
                f"joint names length ({len(self.joint_names)})"
            )


@dataclass
class ActionServerConfig:
    """
    Configuration for a single action server in the fake gripper.
    """

    action_type: Type
    """
    The ROS2 action type for this server.
    """

    topic: str
    """
    The topic name for this action server.
    """

    open_gripper: bool
    """
    Whether this action opens the gripper (True) or closes it (False).
    """

    goal_handler: Callable[[], object]
    """
    Function that creates the goal result for successful completion.
    """


@dataclass
class FakeGripperConfig:
    """
    Configuration for the fake gripper action server.
    """

    node_name: str
    """
    Name for the ROS2 node.
    """

    gripper_connector: GripperConnector
    """
    Configuration for the gripper joints and their positions.
    """

    action_servers: list[ActionServerConfig]
    """
    List of action servers to create for this gripper.
    """

    joint_state_topic: str = "/joint_states"
    """
    Topic to publish joint states on.
    """

    grip_device_state: int = field(
        default=WPGGripperDeviceState.HOLDING,
    )
    """
    Device state reported after grip and flexgrip actions.

    Defaults to :attr:`~WPGGripperDeviceState.HOLDING` (5).
    """

    release_device_state: int = field(
        default=WPGGripperDeviceState.RELEASED,
    )
    """
    Device state reported after release and flexrelease actions.

    Defaults to :attr:`~WPGGripperDeviceState.RELEASED` (3).
    """

    part_detector: Optional[Callable[[], bool]] = field(default=None)
    """
    Optional callable consulted before a grip result is created.

    If set, grip actions call it to determine whether a part was encountered. Returns
    ``True`` if a part is detected (→ ``HOLDING``), ``False`` if no part is found (→
    ``NO_PART``). If ``None``, grip always assumes a part is present.
    """

    devstate_service_name: Optional[str] = field(default=None)
    """
    Name of the devstate ROS2 service.

    If provided, a service server is created that returns the gripper's current device
    state on demand (same interface as the real driver).
    """


class FakeGripperActionServer(ABC):
    """
    Generic base class for fake gripper action servers.

    This class provides the infrastructure for creating fake action servers that
    simulate gripper behavior by publishing joint states to the semantic digital twin.
    It's designed to be extensible for different gripper types and action interfaces.
    """

    def __init__(self, config: FakeGripperConfig):
        self.config = config
        self.current_joint_positions = dict(
            zip(
                config.gripper_connector.joint_names,
                config.gripper_connector.open_positions,
            )
        )
        self._action_servers: dict[str, ActionServer] = {}
        self._joint_state_pub = None
        self._shutdown_callback = None
        self._current_device_state: int = config.release_device_state
        self._devstate_server = None

    def start(self):
        """
        Start the fake gripper action server node and all action servers.
        """
        rospy.init_node(self.config.node_name)
        logger.info(f"Starting fake gripper action server: {self.config.node_name}")

        self._create_joint_state_publisher()
        self._create_action_servers()
        self._create_devstate_service()

    def stop(self):
        """
        Stop the action servers and clean up resources.
        """
        logger.info(f"Stopping fake gripper action server: {self.config.node_name}")
        for topic, server in self._action_servers.items():
            server.destroy()
            logger.debug(f"Destroyed action server: {topic}")

        if self._devstate_server is not None:
            self._devstate_server.destroy()
            logger.debug(
                f"Destroyed devstate service: {self.config.devstate_service_name}"
            )

        if self._joint_state_pub is not None:
            self._joint_state_pub.destroy()

        rospy.shutdown()

    def _create_joint_state_publisher(self):
        """
        Create a publisher for joint states on the configured topic.
        """
        self._joint_state_pub = rospy.node.create_publisher(
            JointState, self.config.joint_state_topic, 10
        )
        logger.info(
            f"Created joint state publisher on: {self.config.joint_state_topic}"
        )

    def _create_action_servers(self):
        """
        Create action servers based on the configuration.
        """
        for action_config in self.config.action_servers:
            self._create_single_action_server(action_config)

    def _create_single_action_server(self, action_config: ActionServerConfig):
        """
        Create a single action server with the given configuration.

        :param action_config: Configuration for this action server.
        """
        action_server = ActionServer(
            rospy.node,
            action_config.action_type,
            action_config.topic,
            execute_callback=lambda goal_handle: self._handle_action_goal(
                goal_handle, action_config
            ),
        )

        self._action_servers[action_config.topic] = action_server
        logger.info(f"Created action server: {action_config.topic}")

    def _handle_action_goal(self, goal_handle, action_config: ActionServerConfig):
        """
        Handle incoming action goals by updating joint states and completing
        immediately.

        :param goal_handle: The ROS2 action server goal handle.
        :param action_config: Configuration for this action.
        :return: Action result indicating success.
        """
        goal = goal_handle.request
        positions = (
            self.config.gripper_connector.open_positions
            if action_config.open_gripper
            else self.config.gripper_connector.closed_positions
        )

        self._update_joint_states(positions)

        result = action_config.goal_handler()

        goal_handle.succeed()
        logger.info(f"Action '{action_config.topic}' completed successfully")

        return result

    def _update_joint_states(self, positions: list[float]):
        """
        Update and publish the current joint states.

        :param positions: List of positions corresponding to joint_names.
        """
        for joint_name, position in zip(
            self.config.gripper_connector.joint_names, positions
        ):
            self.current_joint_positions[joint_name] = position

        if self._joint_state_pub is not None:
            self._publish_joint_states()
        else:
            logger.warning("Joint state publisher not initialized, skipping publish")

    def _publish_joint_states(self):
        """
        Publish the current joint states to the joint state topic.
        """
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.node.get_clock().now().to_msg()
        joint_state_msg.name = list(self.current_joint_positions.keys())
        joint_state_msg.position = list(self.current_joint_positions.values())
        joint_state_msg.velocity = [0.0] * len(joint_state_msg.name)
        joint_state_msg.effort = [0.0] * len(joint_state_msg.name)

        self._joint_state_pub.publish(joint_state_msg)
        logger.debug(
            f"Published joint states: {dict(zip(joint_state_msg.name, joint_state_msg.position))}"
        )

    def _create_devstate_service(self):
        """
        Create a devstate service server if a service name is configured.

        The service mimics the real GRIPLINK driver's devstate endpoint, returning the
        gripper's current internal device state.
        """
        if self.config.devstate_service_name is None:
            logger.info("No devstate service name configured, skipping")
            return
        if not GIPLINK_SRV_AVAILABLE:
            logger.warning(
                "griplink_interfaces srv not available; skipping devstate service"
            )
            return
        self._devstate_server = rospy.node.create_service(
            Devstate,
            self.config.devstate_service_name,
            self._handle_devstate_request,
        )
        logger.info(f"Created devstate service at {self.config.devstate_service_name}")

    def _handle_devstate_request(self, request, response):
        """
        Handle a devstate service request.

        :param request: The service request (port field is ignored).
        :param response: The service response to populate.
        :return: The populated response with the current device state.
        """
        response.status = 0
        response.message = "Success"
        response.state = self._current_device_state
        return response


class WPGFakeGripperActionServer(FakeGripperActionServer):
    """
    Fake action server for WPG-300 grippers used on DAiSy.

    Provides action servers for Grip, Release, Flexgrip, and Flexrelease actions,
    simulating gripper behavior by updating joint states in the semantic digital twin.
    """

    def __init__(self, arm_name: str):
        if not GIPLINK_AVAILABLE:
            raise ImportError(
                "griplink_interfaces package is required for WPGFakeGripperActionServer"
            )

        config = self._create_wpg_config(arm_name)
        super().__init__(config)

    def _create_wpg_config(self, arm_name: str) -> FakeGripperConfig:
        """
        Create configuration for the WPG gripper action server.

        :param arm_name: Either 'left' or 'right' gripper.
        :return: Configuration for the WPG gripper.
        """
        if arm_name not in ["left", "right"]:
            raise ValueError(f"Invalid arm name: {arm_name}. Must be 'left' or 'right'")

        gripper_connector = GripperConnector(
            joint_names=[
                f"{arm_name}_gripper_finger_joint",
                f"{arm_name}_gripper_right_finger_joint",
            ],
            open_positions=[0.0, 0.0],
            closed_positions=[0.04, 0.04],
        )

        action_server_configs = [
            ActionServerConfig(
                action_type=Grip,
                topic=f"/{arm_name}_gripper/grip",
                open_gripper=False,
                goal_handler=self._create_grip_result,
            ),
            ActionServerConfig(
                action_type=Release,
                topic=f"/{arm_name}_gripper/release",
                open_gripper=True,
                goal_handler=self._create_release_result,
            ),
            ActionServerConfig(
                action_type=Flexgrip,
                topic=f"/{arm_name}_gripper/flexgrip",
                open_gripper=False,
                goal_handler=self._create_flexgrip_result,
            ),
            ActionServerConfig(
                action_type=Flexrelease,
                topic=f"/{arm_name}_gripper/flexrelease",
                open_gripper=True,
                goal_handler=self._create_flexrelease_result,
            ),
        ]

        return FakeGripperConfig(
            node_name=f"wpg_senseplan_gripper_{arm_name}_fake",
            gripper_connector=gripper_connector,
            action_servers=action_server_configs,
            joint_state_topic="/joint_states",
            devstate_service_name=f"/{arm_name}_gripper/devstate",
        )

    def _resolved_grip_device_state(self) -> int:
        """
        Determine the device state for a grip action.

        Consults the :attr:`~FakeGripperConfig.part_detector` if configured, otherwise
        defaults to :attr:`~FakeGripperConfig.grip_device_state`.

        :return: The device state value.
        """
        if self.config.part_detector is not None:
            return (
                WPGGripperDeviceState.HOLDING
                if self.config.part_detector()
                else WPGGripperDeviceState.NO_PART
            )
        return WPGGripperDeviceState.HOLDING

    def _create_grip_result(self) -> Grip.Result:
        """
        Create a result message for the Grip action.

        :return: Result with success status (0) and device_state set to HOLDING (or
            NO_PART if :attr:`~FakeGripperConfig.part_detector` returns ``False``).
        """
        result = Grip.Result()
        result.status = 0
        result.message = "Success"
        self._current_device_state = self._resolved_grip_device_state()
        result.device_state = self._current_device_state
        return result

    def _create_release_result(self) -> Release.Result:
        """
        Create a result message for the Release action.

        :return: Result with success status (0) and device_state set to RELEASED.
        """
        result = Release.Result()
        result.status = 0
        result.message = "Success"
        self._current_device_state = self.config.release_device_state
        result.device_state = self._current_device_state
        return result

    def _create_flexgrip_result(self) -> Flexgrip.Result:
        """
        Create a result message for the Flexgrip action.

        :return: Result with success status (0) and device_state set to HOLDING (or
            NO_PART if :attr:`~FakeGripperConfig.part_detector` returns ``False``).
        """
        result = Flexgrip.Result()
        result.status = 0
        result.message = "Success"
        self._current_device_state = self._resolved_grip_device_state()
        result.device_state = self._current_device_state
        return result

    def _create_flexrelease_result(self) -> Flexrelease.Result:
        """
        Create a result message for the Flexrelease action.

        :return: Result with success status (0) and device_state set to RELEASED.
        """
        result = Flexrelease.Result()
        result.status = 0
        result.message = "Success"
        self._current_device_state = self.config.release_device_state
        result.device_state = self._current_device_state
        return result
