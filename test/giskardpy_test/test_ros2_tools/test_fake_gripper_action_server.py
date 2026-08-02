"""
Tests for the fake gripper action server implementation.

This module contains comprehensive tests for the generic FakeGripperActionServer base
class and the WPG-specific WPGFakeGripperActionServer implementation.
"""

import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

import pytest
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

try:
    from griplink_interfaces.action import Grip, Release, Flexgrip, Flexrelease

    GIPLINK_AVAILABLE = True
except ImportError:
    GIPLINK_AVAILABLE = False

from giskardpy.middleware.ros2.scripts.tools.fake_gripper_action_server import (
    FakeGripperActionServer,
    WPGFakeGripperActionServer,
    FakeGripperConfig,
    GripperConnector,
    ActionServerConfig,
)
from giskardpy.middleware.ros2 import rospy


class TestAction:
    """
    Mock action type for testing generic functionality.
    """

    class Goal:
        def __init__(self):
            pass

    class Result:
        def __init__(self):
            self.status = 0
            self.message = "Test success"


@pytest.fixture()
def simple_gripper_config():
    """
    Create a simple gripper configuration for testing.
    """
    return GripperConnector(
        joint_names=["test_finger", "test_finger_right"],
        open_positions=[0.0, 0.0],
        closed_positions=[0.05, 0.05],
    )


@pytest.fixture()
def test_action_config(simple_gripper_config):
    """
    Create a fake gripper config with mock action for testing.
    """

    def test_goal_handler():
        result = TestAction.Result()
        return result

    action_config = ActionServerConfig(
        action_type=TestAction,
        topic="/test_gripper/open",
        open_gripper=True,
        goal_handler=test_goal_handler,
    )

    return FakeGripperConfig(
        node_name="test_fake_gripper",
        gripper_connector=simple_gripper_config,
        action_servers=[action_config],
        joint_state_topic="/test_joint_states",
    )


@pytest.fixture()
def wpg_skip_condition():
    """
    Skip WPG tests if griplink interfaces are not available.
    """
    if not GIPLINK_AVAILABLE:
        pytest.skip("griplink_interfaces not available")


class TestFakeGripperActionServer:
    """
    Unit tests for the generic FakeGripperActionServer functionality.
    """

    def test_initialization_with_valid_config(self, test_action_config):
        """
        Test that fake gripper action server initializes correctly with valid config.
        """
        server = FakeGripperActionServer(test_action_config)

        assert server.config == test_action_config
        assert server.config.node_name == "test_fake_gripper"
        assert len(server.config.action_servers) == 1
        assert len(server.current_joint_positions) == 2

    def test_joint_state_publisher_creation(self, test_action_config, init_rospy):
        """
        Test that joint state publisher is created properly.
        """
        server = FakeGripperActionServer(test_action_config)
        server._create_joint_state_publisher()

        assert server._joint_state_pub is not None
        assert server._joint_state_pub.topic_name == "/test_joint_states"

        server._joint_state_pub.destroy()

    def test_single_action_server_configuration(self, test_action_config, init_rospy):
        """
        Test that single action server configuration is correctly set up.
        """
        server = FakeGripperActionServer(test_action_config)
        action_config = test_action_config.action_servers[0]

        assert action_config.topic == "/test_gripper/open"
        assert action_config.open_gripper == True
        assert action_config.action_type == TestAction
        assert callable(action_config.goal_handler)

    def test_multiple_action_servers_configuration(self, init_rospy):
        """
        Test that multiple action servers can be configured for one gripper.
        """
        from giskardpy.middleware.ros2.scripts.tools.fake_gripper_action_server import (
            FakeGripperConfig,
            ActionServerConfig,
            FakeGripperActionServer,
            GripperConnector,
        )

        def open_handler():
            result = TestAction.Result()
            result.status = 1
            result.message = "Opened"
            return result

        def close_handler():
            result = TestAction.Result()
            result.status = 2
            result.message = "Closed"
            return result

        simple_gripper_config = GripperConnector(
            joint_names=["finger"],
            open_positions=0.0,
            closed_positions=0.05,
        )

        action_configs = [
            ActionServerConfig(
                action_type=TestAction,
                topic="/gripper/open",
                open_gripper=True,
                goal_handler=open_handler,
            ),
            ActionServerConfig(
                action_type=TestAction,
                topic="/gripper/close",
                open_gripper=False,
                goal_handler=close_handler,
            ),
        ]

        test_config = FakeGripperConfig(
            node_name="multi_action_gripper",
            gripper_connector=simple_gripper_config,
            action_servers=action_configs,
        )

        server = FakeGripperActionServer(test_config)

        assert len(server.config.action_servers) == 2
        assert server.config.action_servers[0].topic == "/gripper/open"
        assert server.config.action_servers[1].topic == "/gripper/close"
        assert server.config.action_servers[0].open_gripper == True
        assert server.config.action_servers[1].open_gripper == False

    def test_joint_state_updates_open_gripper(self, test_action_config, init_rospy):
        """
        Test that joint positions update correctly for open gripper state.
        """
        server = FakeGripperActionServer(test_action_config)
        open_positions = [0.0, 0.0]

        server._update_joint_states(open_positions)

        assert server.current_joint_positions["test_finger"] == 0.0
        assert server.current_joint_positions["test_finger_right"] == 0.0

    def test_joint_state_updates_closed_gripper(self, test_action_config, init_rospy):
        """
        Test that joint positions update correctly for closed gripper state.
        """
        server = FakeGripperActionServer(test_action_config)
        closed_positions = [0.05, 0.05]

        server._update_joint_states(closed_positions)

        assert server.current_joint_positions["test_finger"] == 0.05
        assert server.current_joint_positions["test_finger_right"] == 0.05

    def test_joint_state_publishing(self, test_action_config, init_rospy):
        """
        Test that joint states are published correctly.
        """
        server = FakeGripperActionServer(test_action_config)
        server._create_joint_state_publisher()

        mock_publisher = MagicMock()
        server._joint_state_pub = mock_publisher

        server._update_joint_states([0.0, 0.05])
        server._publish_joint_states()

        assert mock_publisher.publish.called
        published_msg = mock_publisher.publish.call_args[0][0]
        assert isinstance(published_msg, JointState)
        assert "test_finger" in published_msg.name
        assert "test_finger_right" in published_msg.name

        joint_positions = dict(zip(published_msg.name, published_msg.position))
        assert joint_positions["test_finger"] == 0.0
        assert joint_positions["test_finger_right"] == 0.05

        server._joint_state_pub.destroy()

    def test_stop_and_cleanup(self, test_action_config, init_rospy):
        """
        Test that stop method properly cleans up resources.
        """
        server = FakeGripperActionServer(test_action_config)

        mock_action_server = MagicMock()
        mock_ws_publisher = MagicMock()

        server._action_servers["/test_gripper/open"] = mock_action_server
        server._joint_state_pub = mock_ws_publisher

        server.stop()

        assert mock_action_server.destroy.called
        assert mock_ws_publisher.destroy.called

    def test_invalid_config_raises_errors(self):
        """
        Test that invalid gripper configuration raises appropriate errors.
        """
        with pytest.raises(ValueError):
            GripperConnector(
                joint_names=["finger1", "finger2"],
                open_positions=[0.0],
                closed_positions=[0.05, 0.05],
            )


class TestWPGFakeGripperActionServer:
    """
    Unit tests for the WPG-specific fake gripper action server.
    """

    def test_wpg_left_gripper_initialization(self, wpg_skip_condition, init_rospy):
        """
        Test that WPG left gripper initializes correctly.
        """
        server = WPGFakeGripperActionServer("left")

        assert server.config.node_name == "wpg_senseplan_gripper_left_fake"
        assert len(server.config.action_servers) == 4
        assert "left_gripper_finger_joint" in server.current_joint_positions
        assert "left_gripper_right_finger_joint" in server.current_joint_positions

    def test_wpg_right_gripper_initialization(self, wpg_skip_condition, init_rospy):
        """
        Test that WPG right gripper initializes correctly.
        """
        server = WPGFakeGripperActionServer("right")

        assert server.config.node_name == "wpg_senseplan_gripper_right_fake"
        assert len(server.config.action_servers) == 4
        assert "right_gripper_finger_joint" in server.current_joint_positions
        assert "right_gripper_right_finger_joint" in server.current_joint_positions

    def test_wpg_invalid_arm_name_raises_error(self, wpg_skip_condition, init_rospy):
        """
        Test that invalid arm name raises ValueError.
        """
        with pytest.raises(ValueError):
            WPGFakeGripperActionServer("invalid_arm")

    def test_wpg_action_servers_topics(self, wpg_skip_condition, init_rospy):
        """
        Test that WPG gripper creates the expected action server topics.
        """
        server = WPGFakeGripperActionServer("left")

        expected_topics = [
            "/left_gripper/grip",
            "/left_gripper/release",
            "/left_gripper/flexgrip",
            "/left_gripper/flexrelease",
        ]

        server._create_action_servers()

        for topic in expected_topics:
            assert topic in server._action_servers

        for action_server in server._action_servers.values():
            action_server.destroy()

    def test_wpg_grip_action_result_creation(self, wpg_skip_condition, init_rospy):
        """
        Test that grip action result is created with correct status.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_grip_result()

        assert result.status == 0
        assert result.message == "Success"

    def test_wpg_release_action_result_creation(self, wpg_skip_condition, init_rospy):
        """
        Test that release action result is created with correct status.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_release_result()

        assert result.status == 0
        assert result.message == "Success"

    def test_wpg_flexgrip_action_result_creation(self, wpg_skip_condition, init_rospy):
        """
        Test that flexgrip action result is created with correct status.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_flexgrip_result()

        assert result.status == 0
        assert result.message == "Success"

    def test_wpg_flexrelease_action_result_creation(
        self, wpg_skip_condition, init_rospy
    ):
        """
        Test that flexrelease action result is created with correct status.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_flexrelease_result()

        assert result.status == 0
        assert result.message == "Success"

    def test_wpg_joint_positions_open_state(self, wpg_skip_condition, init_rospy):
        """
        Test that WPG gripper initializes with correct open joint positions.
        """
        server = WPGFakeGripperActionServer("left")

        open_positions = server.config.gripper_connector.open_positions
        assert open_positions == [0.0, 0.0]

        assert server.current_joint_positions["left_gripper_finger_joint"] == 0.0
        assert server.current_joint_positions["left_gripper_right_finger_joint"] == 0.0

    def test_wpg_joint_positions_closed_state(self, wpg_skip_condition, init_rospy):
        """
        Test that WPG gripper sets correct closed joint positions.
        """
        server = WPGFakeGripperActionServer("left")
        closed_positions = server.config.gripper_connector.closed_positions

        server._update_joint_states(closed_positions)

        assert server.current_joint_positions["left_gripper_finger_joint"] == 0.04
        assert server.current_joint_positions["left_gripper_right_finger_joint"] == 0.04

    def test_wpg_right_gripper_joint_names(self, wpg_skip_condition, init_rospy):
        """
        Test that WPG right gripper has correct joint names.
        """
        server = WPGFakeGripperActionServer("right")

        joint_names = server.config.gripper_connector.joint_names
        assert "right_gripper_finger_joint" in joint_names
        assert "right_gripper_right_finger_joint" in joint_names

    def test_grip_result_includes_device_state_holding(
        self, wpg_skip_condition, init_rospy
    ):
        """
        Test that grip result includes device_state set to HOLDING.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_grip_result()

        assert result.status == 0
        assert result.device_state == 5

    def test_release_result_includes_device_state_released(
        self, wpg_skip_condition, init_rospy
    ):
        """
        Test that release result includes device_state set to RELEASED.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_release_result()

        assert result.status == 0
        assert result.device_state == 3

    def test_flexgrip_result_includes_device_state_holding(
        self, wpg_skip_condition, init_rospy
    ):
        """
        Test that flexgrip result includes device_state set to HOLDING.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_flexgrip_result()

        assert result.status == 0
        assert result.device_state == 5

    def test_flexrelease_result_includes_device_state_released(
        self, wpg_skip_condition, init_rospy
    ):
        """
        Test that flexrelease result includes device_state set to RELEASED.
        """
        server = WPGFakeGripperActionServer("left")
        result = server._create_flexrelease_result()

        assert result.status == 0
        assert result.device_state == 3

    def test_internal_state_updated_after_grip(self, wpg_skip_condition, init_rospy):
        """
        Test that _current_device_state is updated after grip/release.
        """
        server = WPGFakeGripperActionServer("left")
        assert server._current_device_state == 3

        server._create_grip_result()
        assert server._current_device_state == 5

        server._create_release_result()
        assert server._current_device_state == 3

    def test_devstate_service_name_configured(self, wpg_skip_condition, init_rospy):
        """
        Test that devstate service name is configured correctly.
        """
        left_server = WPGFakeGripperActionServer("left")
        assert left_server.config.devstate_service_name == "/left_gripper/devstate"

        right_server = WPGFakeGripperActionServer("right")
        assert right_server.config.devstate_service_name == "/right_gripper/devstate"

    def test_part_detector_callback_when_set(self, wpg_skip_condition, init_rospy):
        """
        Test that part_detector callback affects grip device_state.
        """
        from coraplex.querying.gripper_verification import WPGGripperDeviceState

        server = WPGFakeGripperActionServer("left")
        server.config.part_detector = lambda: True
        assert server._resolved_grip_device_state() == WPGGripperDeviceState.HOLDING

        server.config.part_detector = lambda: False
        assert server._resolved_grip_device_state() == WPGGripperDeviceState.NO_PART

    def test_part_detector_none_always_holds(self, wpg_skip_condition, init_rospy):
        """
        Test that without part_detector, grip always assumes HOLDING.
        """
        from coraplex.querying.gripper_verification import WPGGripperDeviceState

        server = WPGFakeGripperActionServer("left")
        assert server.config.part_detector is None
        assert server._resolved_grip_device_state() == WPGGripperDeviceState.HOLDING


class TestFakeGripperIntegration:
    """
    Integration tests for fake gripper action servers.
    """

    def test_action_server_configuration_communication(
        self, test_action_config, init_rospy
    ):
        """
        Test that action server configuration supports communication setup.
        """
        server = FakeGripperActionServer(test_action_config)

        server._create_joint_state_publisher()

        assert server._joint_state_pub is not None
        assert server.config.gripper_connector.joint_names == [
            "test_finger",
            "test_finger_right",
        ]
        assert server.config.gripper_connector.open_positions == [0.0, 0.0]
        assert server.config.gripper_connector.closed_positions == [0.05, 0.05]
        assert server.config.joint_state_topic == "/test_joint_states"

        server._joint_state_pub.destroy()

    def test_multiple_grippers_independent_configuration(self, init_rospy):
        """
        Test that multiple gripper servers can be configured independently.
        """
        simple_config = GripperConnector(
            joint_names=["left_finger"],
            open_positions=0.0,
            closed_positions=0.05,
        )

        def test_handler():
            result = TestAction.Result()
            return result

        left_config = FakeGripperConfig(
            node_name="left_gripper",
            gripper_connector=simple_config,
            action_servers=[
                ActionServerConfig(
                    action_type=TestAction,
                    topic="/left_gripper/action",
                    open_gripper=True,
                    goal_handler=test_handler,
                )
            ],
        )

        simple_config_right = GripperConnector(
            joint_names=["right_finger"],
            open_positions=0.0,
            closed_positions=0.05,
        )

        right_config = FakeGripperConfig(
            node_name="right_gripper",
            gripper_connector=simple_config_right,
            action_servers=[
                ActionServerConfig(
                    action_type=TestAction,
                    topic="/right_gripper/action",
                    open_gripper=True,
                    goal_handler=test_handler,
                )
            ],
        )

        left_server = FakeGripperActionServer(left_config)
        right_server = FakeGripperActionServer(right_config)

        assert left_server.config.node_name != right_server.config.node_name
        assert left_server.current_joint_positions.keys().isdisjoint(
            right_server.current_joint_positions.keys()
        )
        assert "left_finger" in left_server.current_joint_positions
        assert "right_finger" in right_server.current_joint_positions
