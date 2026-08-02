from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.querying.gripper_verification import (
    IsGripperHoldingPart,
    IsGripperNotFullyClosed,
    WPGGripperDeviceState,
)
from coraplex.robot_plans.actions.core.cable_grasp import CableGraspAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import (
    DAiSy,
    DAiSyLeftGripper,
    DAiSyRightGripper,
)
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class TestWPGGripperDeviceState:
    def test_holding_is_value_5(self):
        assert WPGGripperDeviceState.HOLDING == 5

    def test_no_part_is_value_4(self):
        assert WPGGripperDeviceState.NO_PART == 4

    def test_is_int_enum(self):
        assert isinstance(WPGGripperDeviceState.HOLDING, int)


class TestIsGripperHoldingPart:

    @pytest.fixture
    def daisy_gripper(self, daisy_world) -> DAiSyLeftGripper:
        daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
        return daisy.left_arm.end_effector

    @pytest.fixture
    def mock_rclpy(self):
        with patch(
            "coraplex.querying.gripper_verification._rclpy_module", create=True
        ) as mock_rclpy:
            yield mock_rclpy

    @pytest.fixture
    def mock_devstate_srv(self):
        with patch(
            "coraplex.querying.gripper_verification.DevstateSrv", create=True
        ) as mock_srv:
            yield mock_srv

    def test_holding_when_device_state_is_holding(
        self, daisy_gripper, mock_rclpy, mock_devstate_srv
    ):
        mock_node = MagicMock()
        mock_client = MagicMock()
        mock_node.create_client.return_value = mock_client
        mock_client.wait_for_service.return_value = True

        mock_response = MagicMock()
        mock_response.state = WPGGripperDeviceState.HOLDING
        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = mock_response
        mock_client.call_async.return_value = mock_future

        predicate = IsGripperHoldingPart(
            daisy_gripper, ros_node=mock_node, service_name="/test/devstate"
        )
        assert predicate()

    def test_not_holding_when_device_state_is_no_part(
        self, daisy_gripper, mock_rclpy, mock_devstate_srv
    ):
        mock_node = MagicMock()
        mock_client = MagicMock()
        mock_node.create_client.return_value = mock_client
        mock_client.wait_for_service.return_value = True

        mock_response = MagicMock()
        mock_response.state = WPGGripperDeviceState.NO_PART
        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = mock_response
        mock_client.call_async.return_value = mock_future

        predicate = IsGripperHoldingPart(
            daisy_gripper, ros_node=mock_node, service_name="/test/devstate"
        )
        assert not predicate()

    def test_false_when_service_not_available(
        self, daisy_gripper, mock_rclpy, mock_devstate_srv
    ):
        mock_node = MagicMock()
        mock_client = MagicMock()
        mock_node.create_client.return_value = mock_client
        mock_client.wait_for_service.return_value = False

        predicate = IsGripperHoldingPart(
            daisy_gripper, ros_node=mock_node, service_name="/test/devstate"
        )
        assert not predicate()

    def test_false_when_call_times_out(
        self, daisy_gripper, mock_rclpy, mock_devstate_srv
    ):
        mock_node = MagicMock()
        mock_client = MagicMock()
        mock_node.create_client.return_value = mock_client
        mock_client.wait_for_service.return_value = True

        mock_future = MagicMock()
        mock_future.done.return_value = False
        mock_client.call_async.return_value = mock_future

        predicate = IsGripperHoldingPart(
            daisy_gripper, ros_node=mock_node, service_name="/test/devstate"
        )
        assert not predicate()

    def test_false_when_rclpy_not_available(self, daisy_gripper, mock_devstate_srv):
        with patch("coraplex.querying.gripper_verification._rclpy_module", None):
            predicate = IsGripperHoldingPart(
                daisy_gripper, ros_node=MagicMock(), service_name="/test/devstate"
            )
            assert not predicate()

    def test_false_when_ros_node_is_none(self, mock_rclpy, mock_devstate_srv):
        daisy_gripper = MagicMock()
        predicate = IsGripperHoldingPart(
            daisy_gripper, ros_node=None, service_name="/test/devstate"
        )
        assert not predicate()

    def test_derives_service_name_from_left_gripper(
        self, daisy_gripper, mock_rclpy, mock_devstate_srv
    ):
        mock_node = MagicMock()
        mock_client = MagicMock()
        mock_node.create_client.return_value = mock_client
        mock_client.wait_for_service.return_value = True

        mock_response = MagicMock()
        mock_response.state = WPGGripperDeviceState.HOLDING
        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = mock_response
        mock_client.call_async.return_value = mock_future

        predicate = IsGripperHoldingPart(daisy_gripper, ros_node=mock_node)
        assert predicate.service_name == "/left_gripper/devstate"
        assert predicate()
        mock_node.create_client.assert_called_once_with(
            mock_devstate_srv, "/left_gripper/devstate"
        )


class TestIsGripperNotFullyClosed:

    @pytest.fixture
    def daisy_left_gripper(self, daisy_world) -> DAiSyLeftGripper:
        daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
        return daisy.left_arm.end_effector

    @pytest.fixture
    def daisy_right_gripper(self, daisy_world) -> DAiSyRightGripper:
        daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
        return daisy.right_arm.end_effector

    def _set_joint_position(self, end_effector, joint_name, position):
        connection = end_effector._world.get_connection_by_name(joint_name)
        with end_effector._world.modify_world():
            end_effector._world.state[connection.raw_dof.id].position = position

    def test_not_fully_closed_when_position_below_threshold(self, daisy_left_gripper):
        joint_name = "left_gripper_finger_joint"
        self._set_joint_position(daisy_left_gripper, joint_name, 0.02)
        predicate = IsGripperNotFullyClosed(daisy_left_gripper)
        assert predicate()

    def test_fully_closed_when_position_equals_threshold(self, daisy_left_gripper):
        joint_name = "left_gripper_finger_joint"
        self._set_joint_position(daisy_left_gripper, joint_name, 0.04)
        predicate = IsGripperNotFullyClosed(daisy_left_gripper)
        assert not predicate()

    def test_fully_closed_when_position_equals_closed_position_default(
        self, daisy_left_gripper
    ):
        joint_name = "left_gripper_finger_joint"
        self._set_joint_position(daisy_left_gripper, joint_name, 0.038)
        predicate = IsGripperNotFullyClosed(daisy_left_gripper)
        assert not predicate()

    def test_custom_threshold(self, daisy_left_gripper):
        joint_name = "left_gripper_finger_joint"
        self._set_joint_position(daisy_left_gripper, joint_name, 0.01)
        predicate = IsGripperNotFullyClosed(daisy_left_gripper, closed_position=0.005)
        assert not predicate()

    def test_derives_joint_name_from_right_gripper(self, daisy_right_gripper):
        predicate = IsGripperNotFullyClosed(daisy_right_gripper)
        assert predicate._get_gripper_joint_name() == "right_gripper_finger_joint"

    def test_derives_joint_name_from_left_gripper(self, daisy_left_gripper):
        predicate = IsGripperNotFullyClosed(daisy_left_gripper)
        assert predicate._get_gripper_joint_name() == "left_gripper_finger_joint"


class TestCableGraspPostCondition:

    @pytest.fixture
    def daisy_robot_and_context(self, daisy_world):
        daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
        context = Context(daisy_world, daisy)
        return daisy, context

    def test_post_condition_includes_hardware_checks(self, daisy_robot_and_context):
        daisy, context = daisy_robot_and_context
        left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
        right_end_effector = ViewManager.get_end_effector_view(
            Arms.RIGHT, context.robot
        )

        assert isinstance(left_end_effector, DAiSyLeftGripper)
        assert isinstance(right_end_effector, DAiSyRightGripper)

        hanger_body = Body(
            name=PrefixedName("hanger"),
            collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
        )
        with context.world.modify_world():
            hanger_connection = Connection6DoF.create_with_dofs(
                world=context.world,
                parent=context.world.root,
                child=hanger_body,
                name=PrefixedName("hanger_connection"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.3, y=0.0, z=0.1, reference_frame=context.world.root
                ),
            )
            context.world.add_connection(hanger_connection)

            cable_annotation = Cable.create_with_new_body_in_world(
                name=PrefixedName("cable"),
                world=context.world,
                hanging_from=hanger_body,
                length=0.3,
            )

        condition = CableGraspAction.post_condition(
            variables={},
            context=context,
            kwargs={"cable_annotation": cable_annotation},
        )

        assert condition is not None
