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
from coraplex.utils import translate_pose_along_local_axis
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    ConditionType,
    and_,
    or_,
    variable_from,
)
from coraplex.querying.predicates import GripperIsFree
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

    hanger_body: Body
    """
    Body of the cable hanger to hang the cable on.
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

    gripper_width: int = 0.1
    """
    Opening width of the gripper.

    Used as side offset when grasping the scooped up cable.
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

        scoop_poses = self._calculate_scoop_poses(scoop_arm, scoop_end_effector)

        pre_scoop_pose = scoop_poses["pre_scoop_pose"]
        scoop_pose = scoop_poses["scoop_pose"]
        post_scoop_pose = scoop_poses["post_scoop_pose"]
        clear_scoop_pose = scoop_poses["clear_scoop_pose"]
        return_scoop_pose = scoop_poses["return_scoop_pose"]
        pre_free_cable_pose = scoop_poses["pre_free_cable_pose"]
        free_cable_pose = scoop_poses["free_cable_pose"]

        grasp_poses = self._calculate_grasp_poses(grasp_arm, post_scoop_pose)

        grasp_arm_scoop_pose = grasp_poses["grasp_arm_scoop_pose"]
        approach_grasp_pose = grasp_poses["approach_grasp_pose"]
        pre_grasp_pose = grasp_poses["pre_grasp_pose"]
        grasp_pose = grasp_poses["grasp_pose"]

        front_world, side_world, up_world = self._hanger_axes()

        approach_offset = pre_scoop_pose.to_position().to_np()[:3] + front_world * 0.1
        approach_pose = Pose(
            position=Point3(
                x=approach_offset[0],
                y=approach_offset[1],
                z=approach_offset[2],
                reference_frame=self.world.root,
            ),
            orientation=pre_scoop_pose.orientation,
            reference_frame=self.world.root,
        )

        print(f"Approach pose: {approach_pose.to_position()}")
        print(f"Pre-scoop pose: {pre_scoop_pose.to_position()}")
        print(f"Scoop pose: {scoop_pose.to_position()}")
        print(f"Post-scoop pose: {post_scoop_pose.to_position()}")
        print(f"Clear scoop pose: {clear_scoop_pose.to_position()}")
        print(f"Return scoop pose: {return_scoop_pose.to_position()}")
        print(f"Pre free cable pose: {pre_free_cable_pose.to_position()}")
        print(f"Free cable pose: {free_cable_pose.to_position()}")

        print(f"Grasp arm pose, scoop phase: {grasp_arm_scoop_pose.to_position()}")
        print(f"Approach pose, scoop phase: {approach_grasp_pose.to_position()}")
        print(f"Pre-grasp pose: {pre_grasp_pose.to_position()}")
        print(f"Grasp pose: {grasp_pose.to_position()}")

        gripper_offset_pos = (
            post_scoop_pose.to_position().to_np()[:3]
            - front_world * 0.2 * self.approach_direction
        )

        return sequential(
            children=[
                # Open both grippers
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                MoveGripperMotion(motion=GripperState.OPEN, gripper=grasp_arm),
                ParkArmsAction(arm=scoop_arm),
                # Move scoop arm closer to hanger and rotate already
                MoveToolCenterPointMotion(
                    approach_pose,
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
                    scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=scoop_arm,
                    grip_position=0,
                    grip_speed=150,
                ),
                # Attach the cable to the scoop arm
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=scoop_end_effector.tool_frame,
                ),
                # Move the scoop arm to the post-scoop position, actually scoop cable
                MoveToolCenterPointMotion(
                    post_scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move grasp arm closer to hanger but keep previous orientation
                MoveToolCenterPointMotion(
                    approach_grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move grasp arm in pre-grasp position
                MoveToolCenterPointMotion(
                    pre_grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Close grasp arm gripper halfway
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=grasp_arm,
                    grip_position=70,
                    grip_speed=300,
                    grip_acceleration=2000,
                ),
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXOPEN,
                    gripper=grasp_arm,
                    grip_position=75,
                    grip_speed=300,
                    grip_acceleration=2000,
                ),
                # Move the grasp arm to the grasp position
                MoveToolCenterPointMotion(
                    grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Close gripper of grasp arm to grasp the cable
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=grasp_arm,
                    grip_position=0,
                    grip_force=180,
                ),
                # Attach the cable to the grasp arm
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=grasp_end_effector.tool_frame,
                ),
                # Clear area with scoop arm
                MoveToolCenterPointMotion(
                    clear_scoop_pose, scoop_arm, movement_type=MovementType.CARTESIAN
                ),
                # Roll scoop arm back
                MoveToolCenterPointMotion(
                    Pose(
                        position=clear_scoop_pose.to_position(),
                        orientation=clear_scoop_pose.to_quaternion().multiply(
                            Quaternion.from_rpy(0, 0, pi / 2)
                        ),
                        reference_frame=self.world.root,
                    ),
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Open scoop arm gripper
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                # Move scoop arm to return position
                MoveToolCenterPointMotion(
                    return_scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move grasp arm slightly up
                MoveToolCenterPointMotion(
                    translate_pose_along_local_axis(
                        pose=grasp_pose, axis=[0, 1, 0], distance=-0.07
                    ),
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    pre_free_cable_pose, scoop_arm, movement_type=MovementType.CARTESIAN
                ),
                MoveToolCenterPointMotion(
                    free_cable_pose, scoop_arm, movement_type=MovementType.CARTESIAN
                ),
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=scoop_arm,
                    grip_position=0,
                    grip_force=90,
                    grip_speed=120,
                ),
                MoveToolCenterPointMotion(
                    pre_free_cable_pose, scoop_arm, movement_type=MovementType.CARTESIAN
                ),
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                # Move grasp arm slightly up
                MoveToolCenterPointMotion(
                    translate_pose_along_local_axis(
                        pose=grasp_pose, axis=[0, 1, 0], distance=0.07
                    ),
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
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
    ) -> dict[str, Pose]:
        """
        Calculate the pre-scoop, scoop-end and post-scoop poses for the scooping arm.

        The pre-scoop pose positions the gripper in front of the cable and below the
        hanger, oriented such that the gripper faces the cable. The scoop-end pose moves
        toward the hanging point so the cable is captured between the fingers. The post-
        scoop pose moves the gripper sideways to scoop the cable.
        """
        poses = {}

        front_world, side_world, up_world = self._hanger_axes()
        side_sign = 1.0 if scoop_arm == Arms.RIGHT else -1.0

        hanging_pos = self._hanging_point_position().to_np()

        scoop_orientation = self._scoop_gripper_orientation(
            side_world * side_sign, front_world, z_rotation=pi
        )

        scoop_pos = (
            hanging_pos[:3]
            - front_world * (self.approach_sign * self.front_offset)
            - up_world * self.down_offset
        )
        scoop_pose = Pose(
            position=Point3(
                x=scoop_pos[0],
                y=scoop_pos[1],
                z=scoop_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        poses["scoop_pose"] = scoop_pose

        # Calculate pre-scoop pose, that's right in front of the cable hanger
        pre_scoop_pos = scoop_pos[:3] - front_world * (self.approach_sign * 0.05)

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

        poses["pre_scoop_pose"] = pre_scoop_pose

        # Move to side to scoop up cable, at the same time roll gripper around z axis
        post_scoop_pos = (
            hanging_pos[:3]
            + front_world * self.front_offset
            - side_world * (self.side_offset * side_sign)
            - up_world * self.down_offset
        )

        post_scoop_pose = Pose(
            position=Point3(
                x=post_scoop_pos[0],
                y=post_scoop_pos[1],
                z=post_scoop_pos[2],
                reference_frame=self.world.root,
            ),
            # Roll turn gripper up/down
            # Pitch turns the gripper in the finger plane
            # Yaw rotates the gripper around z (forward pointing axis)
            orientation=scoop_orientation.multiply(
                Quaternion.from_rpy(0, 0, -(pi / 2))
            ),
            reference_frame=self.world.root,
        )

        poses["post_scoop_pose"] = post_scoop_pose

        # Move scoop arm a bit to clear area and get out of the way
        clear_scoop_pos = (
            post_scoop_pos[:3]
            - front_world * (0.05 * self.approach_sign)
            - up_world * 0.1
        )

        clear_scoop_pose = Pose(
            position=Point3(
                x=clear_scoop_pos[0],
                y=clear_scoop_pos[1],
                z=clear_scoop_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=scoop_orientation.multiply(
                Quaternion.from_rpy(0, 0, -(pi / 2))
            ),
            reference_frame=self.world.root,
        )
        poses["clear_scoop_pose"] = clear_scoop_pose

        # Roll scoop gripper back and move down
        return_scoop_pos = (
            pre_scoop_pos[:3]
            - front_world * (0.2 * self.approach_sign)
            + up_world * (0.1 - 0.0477)
        )
        return_scoop_pose = Pose(
            position=Point3(
                x=return_scoop_pos[0],
                y=return_scoop_pos[1],
                z=return_scoop_pos[2],
                reference_frame=self.world.root,
            ),
            # Roll turn gripper up/down
            # Pitch turns the gripper in the finger plane
            # Yaw rotates the gripper around z (forward pointing axis)
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        poses["return_scoop_pose"] = return_scoop_pose

        pre_free_cable_pos = (  # TODO: check these values again
            hanging_pos[:3]
            - front_world * (0.2 * self.approach_sign)
            + up_world * (-0.015)
        )
        pre_free_cable_pose = Pose(
            position=Point3(
                x=pre_free_cable_pos[0],
                y=pre_free_cable_pos[1],
                z=pre_free_cable_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        poses["pre_free_cable_pose"] = pre_free_cable_pose

        free_cable_pos = (
            hanging_pos[:3]
            - front_world * (-0.03 * self.approach_sign)
            + up_world * (-0.015)
        )
        free_cable_pose = Pose(
            position=Point3(
                x=free_cable_pos[0],
                y=free_cable_pos[1],
                z=free_cable_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        poses["free_cable_pose"] = free_cable_pose

        return poses

    def _calculate_grasp_poses(
        self, grasp_arm: Arms, post_scoop_pose: Pose
    ) -> dict[str, Pose]:

        poses = {}

        front_world, side_world, up_world = self._hanger_axes()
        side_sign = 1.0 if grasp_arm == Arms.LEFT else -1.0

        grasp_end_effector = ViewManager.get_end_effector_view(grasp_arm, self.robot)

        # Grasp arm pose during the scoop phase to move the arm out of the way
        current_grasp_arm_transform = grasp_end_effector.tool_frame.global_transform
        current_grasp_arm_position = current_grasp_arm_transform.to_position()
        grasp_arm_scoop_pos = current_grasp_arm_position.to_np()[:3] - side_world * (
            0.15 * side_sign
        )
        current_grasp_arm_orientation = current_grasp_arm_transform.to_quaternion()
        grasp_arm_scoop_pose = Pose(
            Point3(
                x=grasp_arm_scoop_pos[0],
                y=grasp_arm_scoop_pos[1],
                z=grasp_arm_scoop_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=current_grasp_arm_orientation,
            reference_frame=self.world.root,
        )

        poses["grasp_arm_scoop_pose"] = grasp_arm_scoop_pose

        # Approach pose of the grasp arm, same orientation as parked but closer to gripper already
        approach_grasp_pos = (
            post_scoop_pose.to_position().to_np()[:3]
            - side_world * (0.4 * side_sign)
            - up_world * (0.1)
        )
        approach_grasp_pose = Pose(
            position=Point3(
                x=approach_grasp_pos[0],
                y=approach_grasp_pos[1],
                z=approach_grasp_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=current_grasp_arm_orientation,
            reference_frame=self.world.root,
        )

        poses["approach_grasp_pose"] = approach_grasp_pose

        # Pre-grasp position is the post-scoop position with a small offset to the side and below the scoop gripper
        pre_grasp_pos = (
            post_scoop_pose.to_position().to_np()[:3]
            - side_world * (0.2 * side_sign)
            + up_world * (0.1)
        )
        pre_grasp_orientation = self._grasp_gripper_orientation(
            side_world * side_sign,
            front_world,
            pitch_angle=0.7854,
            z_rotation=pi,
        )
        grasp_orientation = self._grasp_gripper_orientation(
            side_world * side_sign,
            front_world,
            pitch_angle=0.7854,
            z_rotation=pi,
        )
        pre_grasp_pose = Pose(
            position=Point3(
                x=pre_grasp_pos[0],
                y=pre_grasp_pos[1],
                z=pre_grasp_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=pre_grasp_orientation,
            reference_frame=self.world.root,
        )

        poses["pre_grasp_pose"] = pre_grasp_pose

        grasp_pos = (
            post_scoop_pose.to_position().to_np()[:3]
            - front_world * (0.01 * self.approach_sign)
            - side_world * (0.014 * side_sign)  # smaller number is to the left
            + up_world * (0.04)
        )
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

        poses["grasp_pose"] = grasp_pose

        return poses

    def _scoop_gripper_orientation(
        self,
        side_direction: np.ndarray,
        front_direction: np.ndarray,
        z_rotation: float = 0.0,
    ) -> Quaternion:
        """
        Compute the gripper orientation quaternion for the scoop arm.

        The gripper's Z axis (front) faces toward the cable (along
        ``-front_direction``). The Y axis (up) is computed to stay in the plane
        containing ``front_direction`` and world Z.

        :param side_direction: Direction vector from the cable's hanging point toward
            the side the scoop arm approaches from.
        :param front_direction: Forward direction of the cable hanger, used to determine
            which way the gripper faces.
        :param z_rotation: Optional rotation in radians around the gripper's Z (forward)
            axis applied after computing the base orientation.
        """
        return _gripper_orientation_from_z_axis(
            -front_direction,
            side_direction,
            z_rotation,  # TODO: Try out with + pi rotation, also adjust pre scoop pose/post scoop pose
        )

    def _grasp_gripper_orientation(
        self,
        side_direction: np.ndarray,
        front_direction: np.ndarray,
        z_rotation: float = 0.0,
        pitch_angle: float = 0.0,
    ) -> Quaternion:
        """
        Compute the gripper orientation quaternion for the grasp arm.

        The gripper's Z axis faces toward the scoop gripper (along ``-side_direction``):
        when the left gripper grasps, Z points right; when the right gripper grasps, Z
        points left.

        The Y axis (up) is computed to stay in the plane containing ``side_direction``
        and world Z.

        :param side_direction: Direction vector from the cable's hanging point toward
            the side of the approach.
        :param front_direction: Forward direction of the cable hanger, used as fallback
            to determine the X axis.
        :param z_rotation: Optional rotation in radians around the gripper's Z (forward)
            axis applied after computing the base orientation.
        """
        return _gripper_orientation_from_z_axis(
            side_direction,
            front_direction,
            z_rotation,
            pitch_angle=pitch_angle,
        )

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
        print(f"hanger_pos: {hanger_pos}")

        left_tip_pos = (
            left_arm.end_effector.tool_frame.global_transform.to_position().to_np()
        )
        right_tip_pos = (
            right_arm.end_effector.tool_frame.global_transform.to_position().to_np()
        )

        print(f"left_tip_pos: {left_tip_pos}, right_tip_pos: {right_tip_pos}")

        left_distance = float(np.linalg.norm(left_tip_pos - hanger_pos))
        right_distance = float(np.linalg.norm(right_tip_pos - hanger_pos))
        print(f"left_distance: {left_distance}, right_distance: {right_distance}")

        # return Arms.LEFT if left_distance <= right_distance else Arms.RIGHT
        return Arms.RIGHT

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
            and_(
                IsGripperHoldingPart(left_end_effector, ros_node=context.ros_node),
                IsGripperNotFullyClosed(left_end_effector),
            ),
            and_(
                IsGripperHoldingPart(right_end_effector, ros_node=context.ros_node),
                IsGripperNotFullyClosed(right_end_effector),
            ),
        )
