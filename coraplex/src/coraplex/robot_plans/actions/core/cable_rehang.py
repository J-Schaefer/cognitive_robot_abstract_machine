from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from math import pi

import numpy as np
from numpy import dtype, ndarray
from typing_extensions import Any, Dict

from coraplex.alternative_motion_mappings.daisy_motion_mapping import (
    DAiSyFlexGripMotion,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, MovementType
from coraplex.plans.attachment_nodes import AttachNode
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    ConditionType,
    and_,
    or_,
    variable_from,
)
from coraplex.querying.predicates import GripperIsFree, GripperIsNotFree
from coraplex.querying.gripper_verification import (
    IsGripperHoldingPart,
    IsGripperNotFullyClosed,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a[:3], b[:3])


def _normalized(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _rotation_matrix_from_axes(
    x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray
) -> RotationMatrix:
    data = np.eye(4)
    data[:3, 0] = x_axis
    data[:3, 1] = y_axis
    data[:3, 2] = z_axis
    return RotationMatrix(data=data)


def _gripper_orientation_from_z_axis(
    gripper_z_axis: np.ndarray,
    fallback_direction: np.ndarray,
    z_rotation: float = 0.0,
    pitch_angle: float = 0.0,
) -> Quaternion:
    """
    Compute a gripper orientation quaternion from a desired Z axis.

    The frame is built so that the Y axis stays in the plane containing the gripper Z
    axis and world Z. The fallback direction disambiguates the X axis when the gripper Z
    axis is parallel to world Z.

    :param gripper_z_axis: Desired direction for the gripper's Z axis (forward).
    :param fallback_direction: Used to determine the X axis when ``gripper_z_axis`` is
        parallel to the world Z axis.
    :param z_rotation: Optional rotation in radians around the gripper's Z axis applied
        after the base orientation is computed.
    """
    gripper_z = _normalized(gripper_z_axis)
    world_up = np.array([0, 0, 1])

    cross_xz = _cross(world_up, gripper_z)
    if np.linalg.norm(cross_xz) < 1e-6:
        fallback = _cross(world_up, fallback_direction)
        if np.linalg.norm(fallback) < 1e-6:
            gripper_x = np.array([1.0, 0.0, 0.0])
        else:
            gripper_x = _normalized(fallback)
    else:
        gripper_x = _normalized(cross_xz)
    gripper_y = _normalized(_cross(gripper_z, gripper_x))

    rotation_matrix = _rotation_matrix_from_axes(gripper_x, gripper_y, gripper_z)
    quaternion = Quaternion.from_rotation_matrix(rotation_matrix)

    if pitch_angle != 0.0:
        # Axis are z to the front, x to the side of the gripper, so gripper pitch up/down is around the x-axis
        quaternion = quaternion.multiply(Quaternion.from_rpy(pitch_angle, 0.0, 0.0))

    if z_rotation != 0.0:
        quaternion = quaternion.multiply(Quaternion.from_rpy(0.0, 0.0, z_rotation))

    return quaternion


@dataclass
class CableRehangAction(ActionDescription):
    """
    Hangs the cable again to the specified hanger.
    """

    cable_annotation: Cable
    """
    The cable semantic annotation to grasp.
    """

    hanger_body: Body
    """
    Body of the cable hanger to hang the cable on.
    """

    side_offset: float = field(default=0.1)
    """
    Distance in metres to offset the hang arm to the side of the hanging point.
    """

    front_offset: float = field(default=0.05)
    """
    Distance in metres to offset the hang arm in front of the hanging point.
    """

    up_offset: float = field(default=0.12)
    """
    Distance in metres to offset the hang arm above the cable hanger.
    """

    approach_direction: int = 0
    """
    Index of the hanger's local axis that is the front-facing axis.

    0 is the X axis, 1 is the Y axis, 2 is the Z axis. The other two axes form a right-
    handed frame with the front axis.
    """

    approach_sign: int = 1  # TODO redefine type hint to only allow -1 or +1
    """
    Direction the approach axis is pointing.

    If the axis is pointing towards the approach direction the approach_sign is +1, if
    the axis is pointing to the back the approach_sign is -1.
    """

    @property
    def _action_plan(self) -> PlanNode:
        holding_arm = self._determine_holding_arm()
        # TODO: fix the arm selection, or check the distances, sometimes with smaller
        #  difference as if it's choosing the wrong arm
        free_arm = Arms.RIGHT if holding_arm == Arms.LEFT else Arms.LEFT
        print(f"Holding with {holding_arm.name}, Free arm {free_arm.name}")

        holding_end_effector = ViewManager.get_end_effector_view(
            holding_arm, self.robot
        )
        free_end_effector = ViewManager.get_end_effector_view(free_arm, self.robot)

        hang_poses = self._calculate_hang_pose(holding_arm)

        front_world, side_world, up_world = self._hanger_axes()

        approach_hang_pose = hang_poses["approach_hang_pose"]
        pre_hang_pose = hang_poses["pre_hang_pose"]
        hang_pose = hang_poses["hang_pose"]

        print(f"Approach hang pose: {approach_hang_pose.to_position()}")
        print(f"Pre hang pose: {pre_hang_pose.to_position()}")
        print(f"Hang pose: {hang_pose.to_position()}")

        side_sign = 1.0 if free_arm == Arms.LEFT else -1.0
        current_free_arm_transform = free_end_effector.tool_frame.global_transform
        current_free_arm_position = current_free_arm_transform.to_position()
        current_free_arm_orientation = current_free_arm_transform.to_quaternion()
        free_arm_out_of_way_pos = current_free_arm_position.to_np()[:3] - side_world * (
            0.15 * side_sign
        )
        free_arm_out_of_way_pose = Pose(
            Point3(
                x=free_arm_out_of_way_pos[0],
                y=free_arm_out_of_way_pos[1],
                z=free_arm_out_of_way_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=current_free_arm_orientation,
            reference_frame=self.world.root,
        )

        return sequential(
            [
                MoveToolCenterPointMotion(
                    target=free_arm_out_of_way_pose,
                    arm=free_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    target=approach_hang_pose,
                    arm=holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    target=pre_hang_pose,
                    arm=holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    target=hang_pose,
                    arm=holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Attach the cable to the grasp arm
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=self.hanger_body,
                ),
                MoveGripperMotion(gripper=holding_arm, motion=GripperState.OPEN),
                MoveToolCenterPointMotion(
                    target=pre_hang_pose,
                    arm=holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
            ],
        )

    def _determine_holding_arm(self) -> Arms:
        cable_body = self.cable_annotation.root
        left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, self.robot)
        right_end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, self.robot)

        parent = cable_body.parent_kinematic_structure_entity

        if parent == left_end_effector.tool_frame:
            return Arms.LEFT
        elif parent == right_end_effector.tool_frame:
            return Arms.RIGHT

        raise RuntimeError("Cable is not attached to any end effector")

    def _hanger_axes(
        self,
    ) -> tuple[
        ndarray[tuple[int, ...], dtype[Any]],
        ndarray[tuple[int, ...], dtype[Any]],
        ndarray[tuple[int, ...], dtype[Any]],
    ]:
        """
        Return world-frame unit vectors (front, side, up) for the hanger.

        `approach_direction` is the frame axis index the hanger faces along
        (0=X, 1=Y, 2=Z); `approach_sign` is +1/-1 if the front points along
        the positive/negative axis. Up is the frame's +Z. The frame is
        right-handed: front × side = up, i.e., side = up × front.
        """
        hanger_rot = self.cable_annotation.hanging_from.global_transform
        rot_np = np.array(hanger_rot.to_np()[:3, :3], dtype=float)

        front = self.approach_sign * rot_np[:, self.approach_direction]
        up = rot_np[:, 2]  # frame's Z is up
        side = np.cross(up, front)  # guarantees front × side = up

        return front, side, up

    def _calculate_hang_pose(self, holding_arm: Arms) -> dict[str, Pose]:
        """
        Calculate the pose to hang the cable back on the hanger.
        """
        poses = {}

        front_world, side_world, up_world = self._hanger_axes()
        side_sign = 1.0 if holding_arm == Arms.RIGHT else -1.0

        hang_pos = self._hanging_point_position().to_np()
        print(f"Hanging pose: {hang_pos[:3]}")

        hang_orientation = self._hang_gripper_orientation(
            side_world * side_sign, front_world, z_rotation=pi
        )

        # hang_pos = (
        #     hanger_pos[:3]
        #     - front_world
        #     * (
        #         self.front_offset * self.approach_sign
        #     )  # TODO: Check why front world is (-0, -1, -0)
        #     + side_world * (self.side_offset * side_sign)
        #     + up_world * self.up_offset
        # )
        hang_pose = Pose(
            position=Point3(
                x=hang_pos[0],
                y=hang_pos[1],
                z=hang_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=hang_orientation,
            reference_frame=self.world.root,
        )
        poses["hang_pose"] = hang_pose

        pre_hang_pos = hang_pos[:3] - front_world * (0.1 * self.approach_sign)
        pre_hang_pose = Pose(
            position=Point3(
                x=pre_hang_pos[0],
                y=pre_hang_pos[1],
                z=pre_hang_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=hang_orientation,
            reference_frame=self.world.root,
        )

        poses["pre_hang_pose"] = pre_hang_pose

        approach_hang_pos = pre_hang_pos[:3] - front_world * 0.1 * self.approach_sign
        approach_hang_pose = Pose(
            position=Point3(
                x=approach_hang_pos[0],
                y=approach_hang_pos[1],
                z=approach_hang_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=hang_orientation,
            reference_frame=self.world.root,
        )

        poses["approach_hang_pose"] = approach_hang_pose

        return poses

    def _hang_gripper_orientation(
        self,
        side_direction: np.ndarray,
        front_direction: np.ndarray,
        z_rotation: float = 0.0,
    ) -> Quaternion:
        """
        Compute the gripper orientation quaternion for the holding/hang arm.

        The gripper's Z axis (front) faces toward the cable (along
        ``-front_direction``). The Y axis (up) is computed to stay in the plane
        containing ``front_direction`` and world Z.

        :param side_direction: Direction vector from the cable's hanging point toward
            the side the hang arm approaches from.
        :param front_direction: Forward direction of the cable hanger, used to determine
            which way the gripper faces.
        :param z_rotation: Optional rotation in radians around the gripper's Z (forward)
            axis applied after computing the base orientation.
        """
        return _gripper_orientation_from_z_axis(
            -front_direction,
            side_direction,
            z_rotation,
        )

    def _hanging_point_position(self) -> Point3:
        """
        Calculates the position of the cable hanging point in global coordinates.

        This method computes the global position of the hanging point by first
        determining the global axes of the hanger, then calculating an offset
        vector based on the provided front, side, and up offsets. The offset
        is transformed into a local transformation matrix, and the position
        is obtained by applying this transformation to the global transformation
        matrix of the parent body.

        Returns:
            Point3: The global position of the hanging point.
        """
        parent_global = self.hanger_body.global_transform
        front_world, side_world, up_world = self._hanger_axes()

        offset = (
            front_world * self.front_offset
            + side_world * self.side_offset
            + up_world * self.up_offset
        )
        local_offset = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=offset[0],
            y=offset[1],
            z=offset[2],
        )
        return (parent_global @ local_offset).to_position()

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
        right_end_effector = ViewManager.get_end_effector_view(
            Arms.RIGHT, context.robot
        )
        return or_(
            and_(
                GripperIsNotFree(left_end_effector),
                GripperIsFree(right_end_effector),
            ),
            and_(
                GripperIsFree(left_end_effector),
                GripperIsNotFree(right_end_effector),
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
        right_end_effector = ViewManager.get_end_effector_view(
            Arms.RIGHT, context.robot
        )
        cable_body = kwargs["cable_annotation"].root
        return or_(
            is_body_in_gripper(variable_from(cable_body), left_end_effector) > 0.9,
            is_body_in_gripper(variable_from(cable_body), right_end_effector) > 0.9,
            and_(
                IsGripperHoldingPart(left_end_effector, ros_node=context.ros_node),
                IsGripperNotFullyClosed(left_end_effector),
            ),
            and_(
                IsGripperHoldingPart(right_end_effector, ros_node=context.ros_node),
                IsGripperNotFullyClosed(right_end_effector),
            ),
        )
