from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from typing_extensions import Any, Dict

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, MovementType
from coraplex.plans.attachment_nodes import AttachNode
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
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
from coraplex.querying.predicates import GripperIsFree
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


@dataclass
class CableGraspAction(ActionDescription):
    """
    Performs a two-handed grasp of a cable.

    One arm scoops the cable up from its hanging point while keeping the gripper open.
    The other arm then grasps the cable below the scooping gripper. The arm used for
    scooping is chosen dynamically based on which arm can better reach the cable hanging
    point.
    """

    cable_annotation: Cable
    """
    The cable semantic annotation to grasp.
    """

    grasp_offset: float = field(default=0.1)
    """
    Vertical distance in metres between the scooping gripper position and the grasping
    gripper position along the world Z axis.
    """

    side_offset: float = field(default=0.1)
    """
    Distance in metres to offset the scoop arm to the side of the hanging point.
    """

    front_offset: float = field(default=0.05)
    """
    Distance in metres to offset the scoop arm in front of the hanging point.
    """

    down_offset: float = field(default=0.12)
    """
    Distance in metres to offset the scoop arm below the cable hanger.
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
        scoop_arm = self._choose_scoop_arm()
        # TODO: fix the arm selection, or check the distances, sometimes with smaller
        #  difference as if it's choosing the wrong arm
        grasp_arm = Arms.RIGHT if scoop_arm == Arms.LEFT else Arms.LEFT
        print(f"Scooping with {scoop_arm.name}, Grasping with {grasp_arm.name}")

        scoop_end_effector = ViewManager.get_end_effector_view(scoop_arm, self.robot)
        grasp_end_effector = ViewManager.get_end_effector_view(grasp_arm, self.robot)

        pre_scoop_pose, scoop_end_pose, post_scoop_pose = self._calculate_scoop_poses(
            scoop_arm, scoop_end_effector
        )

        pre_grasp_pose, grasp_pose = self._calculate_pre_grasp_pose(
            grasp_arm, post_scoop_pose
        )

        # hanging_position = self._hanging_point_position()
        # grasp_position = Point3(
        #     x=hanging_position.x,
        #     y=hanging_position.y,
        #     z=hanging_position.z - self.grasp_offset,
        #     reference_frame=self.world.root,
        # )
        # grasp_pose = Pose(
        #     position=grasp_position,
        #     orientation=grasp_end_effector.front_facing_orientation,
        #     reference_frame=self.world.root,
        # )

        print(f"Pre-scoop pose: {pre_scoop_pose.to_position()}")
        print(f"Scoop-end pose: {scoop_end_pose.to_position()}")
        print(f"Post-scoop pose: {post_scoop_pose.to_position()}")
        print(f"Pre-grasp pose: {pre_grasp_pose.to_position()}")
        print(f"Grasp pose: {grasp_pose.to_position()}")

        return sequential(
            children=[
                # Open both grippers
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                MoveGripperMotion(motion=GripperState.OPEN, gripper=grasp_arm),
                # TODO: Add approach position in front of the hanging point with about 10 cm distance before approaching
                MoveToolCenterPointMotion(
                    pre_scoop_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.2),
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move the scoop arm to the pre-scoop position
                MoveToolCenterPointMotion(
                    pre_scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move the scoop arm to the scoop-end position
                MoveToolCenterPointMotion(
                    scoop_end_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move the scoop arm to the post-scoop position, actually scoop cable
                MoveToolCenterPointMotion(
                    post_scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move grasp arm the pre-grasp position
                MoveToolCenterPointMotion(
                    pre_grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move the grasp arm to the grasp position
                MoveToolCenterPointMotion(
                    grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Close gripper of grasp arm to grasp the cable
                MoveGripperMotion(motion=GripperState.CLOSE, gripper=grasp_arm),
                # Attach the cable to the grasp arm
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=grasp_end_effector.tool_frame,
                ),
            ],
        )

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

    def _calculate_scoop_poses(
        self, scoop_arm: Arms, scoop_end_effector
    ) -> tuple[Pose, Pose, Pose]:
        """
        Calculate the pre-scoop, scoop-end and post-scoop poses for the scooping arm.

        The pre-scoop pose positions the gripper in front of of the cable and below
        the hanger, oriented such that the gripper faces the cable. The scoop-end pose
        moves toward the hanging point so the cable is captured between the fingers.
        The post-scoop pose moves the gripper sideways to scoop the cable.
        """
        front_world, side_world, up_world = self._hanger_axes()
        side_sign = 1.0 if scoop_arm == Arms.RIGHT else -1.0

        hanging_pos = self._hanging_point_position().to_np()

        pre_scoop_pos = (
            hanging_pos[:3]
            + front_world * (-self.front_offset)
            + side_world * (self.side_offset * side_sign)
            - up_world * self.down_offset
        )
        scoop_orientation = self._scoop_gripper_orientation(
            side_world * side_sign, front_world
        )

        pre_scoop_pose = Pose(
            position=Point3(
                x=pre_scoop_pos[0],
                y=pre_scoop_pos[1],
                z=pre_scoop_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        scoop_end_pos = (
            hanging_pos[:3]
            + front_world * (-self.front_offset)
            - up_world * self.down_offset
        )
        scoop_end_pose = Pose(
            position=Point3(
                x=scoop_end_pos[0],
                y=scoop_end_pos[1],
                z=scoop_end_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        post_scoop_pos = hanging_pos[:3] - side_world * (0.2 * side_sign)

        post_scoop_pose = Pose(
            position=Point3(
                x=post_scoop_pos[0],
                y=post_scoop_pos[1],
                z=post_scoop_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        return pre_scoop_pose, scoop_end_pose, post_scoop_pose

    def _calculate_pre_grasp_pose(self, grasp_arm: Arms, post_scoop_pose: Pose):
        front_world, side_world, up_world = self._hanger_axes()
        side_sign = 1.0 if grasp_arm == Arms.LEFT else -1.0

        # Pre-grasp position is the post-scoop position with a small offset to the side and below the scoop gripper
        pre_grasp_pos = (
            post_scoop_pose.to_position().to_np()[:3]
            - side_world * (0.1 * side_sign)
            - up_world * (0.1)
        )
        grasp_orientation = self._grasp_gripper_orientation(
            side_world * side_sign, front_world
        )

        pre_grasp_pose = Pose(
            position=Point3(
                x=pre_grasp_pos[0],
                y=pre_grasp_pos[1],
                z=pre_grasp_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=grasp_orientation,
            reference_frame=self.world.root,
        )

        grasp_pos = post_scoop_pose.to_position().to_np()[:3] - up_world * (0.1)

        grasp_pose = Pose(
            position=Point3(
                x=grasp_pos[0],
                y=grasp_pos[1],
                z=grasp_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=grasp_orientation,
            reference_frame=self.world.root,
        )

        return pre_grasp_pose, grasp_pose

    def _scoop_gripper_orientation(
        self,
        side_direction: np.ndarray,
        front_direction: np.ndarray,
    ) -> Quaternion:
        """
        Compute the gripper orientation quaternion for the scoop arm.

        The gripper's Z axis (front) faces toward the cable (along ``-front_direction``).
        The Y axis (up) is computed to stay in the plane containing ``front_direction``
        and world Z.
        """
        gripper_z = _normalized(-front_direction)
        world_up = np.array([0, 0, 1])

        cross_xz = _cross(world_up, gripper_z)
        if np.linalg.norm(cross_xz) < 1e-6:
            # gripper_z is parallel to world_up (front_direction is vertical);
            # fall back to side_direction to disambiguate the roll axis.
            fallback = _cross(world_up, side_direction)
            if np.linalg.norm(fallback) < 1e-6:
                # Both front and side are vertical — pick an arbitrary horizontal axis.
                gripper_x = np.array([1.0, 0.0, 0.0])
            else:
                gripper_x = _normalized(fallback)
        else:
            gripper_x = _normalized(cross_xz)
        gripper_y = _normalized(_cross(gripper_z, gripper_x))

        rotation_matrix = _rotation_matrix_from_axes(gripper_x, gripper_y, gripper_z)
        return Quaternion.from_rotation_matrix(rotation_matrix)

    def _grasp_gripper_orientation(
        self,
        side_direction: np.ndarray,
        front_direction: np.ndarray,
    ) -> Quaternion:
        """
        Compute the gripper orientation quaternion for the grasp arm.

        The gripper's Z axis faces toward the scoop gripper (along ``-side_direction``):
        when the left gripper grasps, Z points right; when the right gripper grasps,
        Z points left.

        The Y axis (up) is computed to stay in the plane containing ``side_direction``
        and world Z.
        """
        gripper_z = _normalized(side_direction)
        world_up = np.array([0, 0, 1])

        cross_xz = _cross(world_up, gripper_z)
        if np.linalg.norm(cross_xz) < 1e-6:
            # gripper_z is parallel to world_up (side_direction is vertical);
            # fall back to front_direction to disambiguate the roll axis.
            fallback = _cross(world_up, front_direction)
            if np.linalg.norm(fallback) < 1e-6:
                # Both side and front are vertical — pick an arbitrary horizontal axis.
                gripper_x = np.array([1.0, 0.0, 0.0])
            else:
                gripper_x = _normalized(fallback)
        else:
            gripper_x = _normalized(cross_xz)
        gripper_y = _normalized(_cross(gripper_z, gripper_x))

        rotation_matrix = _rotation_matrix_from_axes(gripper_x, gripper_y, gripper_z)
        return Quaternion.from_rotation_matrix(rotation_matrix)

    def _hanging_point_position(self) -> Point3:
        parent_global = self.cable_annotation.hanging_from.global_transform
        local_offset = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.cable_annotation.mount_offset_x,
            y=self.cable_annotation.mount_offset_y,
            z=self.cable_annotation.height_offset,
        )
        return (parent_global @ local_offset).to_position()

    def _choose_scoop_arm(self) -> Arms:
        left_arm = ViewManager.get_arm_view(Arms.LEFT, self.robot)
        right_arm = ViewManager.get_arm_view(Arms.RIGHT, self.robot)

        hanger_pos = self._hanging_point_position().to_np()

        left_tip_pos = (
            left_arm.end_effector.tool_frame.global_transform.to_position().to_np()
        )
        right_tip_pos = (
            right_arm.end_effector.tool_frame.global_transform.to_position().to_np()
        )

        left_distance = float(np.linalg.norm(left_tip_pos - hanger_pos))
        right_distance = float(np.linalg.norm(right_tip_pos - hanger_pos))

        return Arms.LEFT if left_distance <= right_distance else Arms.RIGHT

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
        right_end_effector = ViewManager.get_end_effector_view(
            Arms.RIGHT, context.robot
        )
        return and_(
            GripperIsFree(left_end_effector),
            GripperIsFree(right_end_effector),
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
        )
