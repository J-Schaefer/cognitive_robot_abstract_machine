from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import pi
from typing import Any

import numpy as np

from coraplex.alternative_motion_mappings.daisy_motion_mapping import (
    DAiSyFlexGripMotion,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, MovementType
from coraplex.plans.attachment_nodes import AttachNode
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.gripper_verification import (
    IsGripperHoldingPart,
    IsGripperNotFullyClosed,
)
from coraplex.querying.predicates import GripperIsFree, GripperIsNotFree
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
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.robots.robot_parts import EndEffector
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

# %% shared geometry helpers


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
    :param pitch_angle: Optional rotation in radians around the gripper's X axis after
        the base orientation is computed.
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
        quaternion = quaternion.multiply(Quaternion.from_rpy(pitch_angle, 0.0, 0.0))

    if z_rotation != 0.0:
        quaternion = quaternion.multiply(Quaternion.from_rpy(0.0, 0.0, z_rotation))

    return quaternion


# %% shared action helpers


def _hanger_axes(
    global_transform: HomogeneousTransformationMatrix,
    approach_direction: int,
    approach_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world-frame unit vectors (front, side, up) for a hanger frame.

    ``approach_direction`` is the frame axis index the hanger faces along
    (0=X, 1=Y, 2=Z); ``approach_sign`` is +1/-1 if the front points along the
    positive/negative axis. Up is the frame's +Z. The frame is right-handed:
    front x side = up, i.e., side = up x front.

    :param global_transform: The global transform of the hanger body.
    :param approach_direction: Index of the hanger's local axis that is the
        front-facing axis.
    :param approach_sign: +1 if the front axis points toward the approach
        direction, -1 if opposite.
    """
    rot_np = np.array(global_transform.to_np()[:3, :3], dtype=float)

    front = approach_sign * rot_np[:, approach_direction]
    up = rot_np[:, 2]
    side = np.cross(up, front)

    return front, side, up


def _determine_holding_arm(cable_body: Body, robot: Any) -> Arms:
    """
    Return the arm that is currently holding the cable body.
    """
    left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, robot)
    right_end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)

    parent = cable_body.parent_kinematic_structure_entity

    if parent == left_end_effector.tool_frame:
        return Arms.LEFT
    elif parent == right_end_effector.tool_frame:
        return Arms.RIGHT

    raise RuntimeError("Cable is not attached to any end effector")


def _attachment_transform(
    end_effector: EndEffector,
) -> HomogeneousTransformationMatrix:
    """
    Compute the transform from the end effector's tool frame to the cable body.

    The cable body's Z axis is aligned with the tool frame's Y axis so that the cable
    cylinder is held correctly between the gripper fingers. The cable body is centered
    at the tool frame origin (TCP).
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.0,
        y=0.0,
        z=0.0,
        roll=-pi / 2,
        reference_frame=end_effector.tool_frame,
    )


# %% pre- and post-condition functions


def _pre_condition_both_grippers_free(
    variables: dict[str, Variable],
    context: Context,
    kwargs: dict[str, Any],
) -> ConditionType:
    left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
    right_end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, context.robot)
    return and_(
        GripperIsFree(left_end_effector),
        GripperIsFree(right_end_effector),
    )


def _pre_condition_one_gripper_free(
    variables: dict[str, Variable],
    context: Context,
    kwargs: dict[str, Any],
) -> ConditionType:
    left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
    right_end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, context.robot)
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


def _post_condition_cable_in_either_gripper(
    variables: dict[str, Variable],
    context: Context,
    kwargs: dict[str, Any],
) -> ConditionType:
    left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
    right_end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, context.robot)
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


def _post_condition_cable_in_both_grippers(
    variables: dict[str, Variable],
    context: Context,
    kwargs: dict[str, Any],
) -> ConditionType:
    left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
    right_end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, context.robot)
    cable_body = kwargs["cable_annotation"].root
    return and_(
        is_body_in_gripper(variable_from(cable_body), left_end_effector) > 0.9,
        is_body_in_gripper(variable_from(cable_body), right_end_effector) > 0.9,
    )


# %% CableGraspAction


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

    approach_sign: int = 1
    """
    Direction the approach axis is pointing.

    If the axis is pointing towards the approach direction the approach_sign is +1, if
    the axis is pointing to the back the approach_sign is -1.
    """

    pre_condition = staticmethod(_pre_condition_both_grippers_free)
    post_condition = staticmethod(_post_condition_cable_in_either_gripper)

    @property
    def _action_plan(self) -> PlanNode:
        scoop_arm = self._choose_scoop_arm()
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

        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )

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

        return sequential(
            children=[
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                MoveGripperMotion(motion=GripperState.OPEN, gripper=grasp_arm),
                ParkArmsAction(arm=scoop_arm),
                MoveToolCenterPointMotion(
                    approach_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    pre_scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.STRAIGHT_TRANSLATION,
                    threshold=0.001,
                ),
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=scoop_arm,
                    grip_position=0,
                    grip_speed=150,
                ),
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=scoop_end_effector.tool_frame,
                    parent_T_connection_expression=_attachment_transform(
                        scoop_end_effector
                    ),
                ),
                MoveToolCenterPointMotion(
                    post_scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                    threshold=0.001,
                ),
                MoveToolCenterPointMotion(
                    approach_grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    pre_grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
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
                MoveToolCenterPointMotion(
                    grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                    threshold=0.001,
                ),
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=grasp_arm,
                    grip_position=0,
                    grip_force=180,
                ),
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=grasp_end_effector.tool_frame,
                    parent_T_connection_expression=_attachment_transform(
                        grasp_end_effector
                    ),
                ),
                MoveToolCenterPointMotion(
                    clear_scoop_pose, scoop_arm, movement_type=MovementType.CARTESIAN
                ),
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
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                MoveToolCenterPointMotion(
                    return_scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    target=Pose(
                        translate_pose_along_local_axis(
                            translate_pose_along_local_axis(
                                pose=grasp_pose,
                                axis=[0, 1, 0],
                                distance=-0.07,
                            ),
                            axis=[1, 0, 0],
                            distance=0.2,
                        ).to_position(),
                        orientation=grasp_pose.to_quaternion().multiply(
                            Quaternion.from_rpy(0, 0, pi / 6)
                        ),
                        reference_frame=self.world.root,
                    ),
                    arm=grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    pre_free_cable_pose, scoop_arm, movement_type=MovementType.CARTESIAN
                ),
                MoveToolCenterPointMotion(
                    free_cable_pose,
                    scoop_arm,
                    movement_type=MovementType.STRAIGHT_TRANSLATION,
                    threshold=0.001,
                ),
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=scoop_arm,
                    grip_position=0,
                    grip_force=30,
                    grip_speed=120,
                ),
                MoveToolCenterPointMotion(
                    pre_free_cable_pose, scoop_arm, movement_type=MovementType.CARTESIAN
                ),
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                MoveToolCenterPointMotion(
                    translate_pose_along_local_axis(
                        pose=grasp_pose, axis=[0, 1, 0], distance=0.07
                    ),
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
            ],
        )

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

        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )
        side_sign = 1.0 if scoop_arm == Arms.RIGHT else -1.0

        hanging_pos = self._hanging_point_position().to_np()

        scoop_orientation = _gripper_orientation_from_z_axis(
            -front_world, side_world * side_sign, z_rotation=pi
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
            orientation=scoop_orientation.multiply(
                Quaternion.from_rpy(0, 0, -(pi / 2))
            ),
            reference_frame=self.world.root,
        )

        poses["post_scoop_pose"] = post_scoop_pose

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
            orientation=scoop_orientation,
            reference_frame=self.world.root,
        )

        poses["return_scoop_pose"] = return_scoop_pose

        pre_free_cable_pos = (
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

        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )
        side_sign = 1.0 if grasp_arm == Arms.LEFT else -1.0

        grasp_end_effector = ViewManager.get_end_effector_view(grasp_arm, self.robot)

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

        pre_grasp_pos = (
            post_scoop_pose.to_position().to_np()[:3]
            - side_world * (0.2 * side_sign)
            + up_world * (0.1)
        )
        pre_grasp_orientation = _gripper_orientation_from_z_axis(
            side_world * side_sign, front_world, z_rotation=pi, pitch_angle=0.7854
        )
        grasp_orientation = _gripper_orientation_from_z_axis(
            side_world * side_sign, front_world, z_rotation=pi, pitch_angle=0.7854
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
            - side_world * (0.014 * side_sign)
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

        return Arms.RIGHT


# %% CableRegraspAction


@dataclass
class CableRegraspAction(ActionDescription):
    """
    Regrasps a cable that is already held by one arm.

    After the initial grasp, one arm holds the cable and the other arm is free. This
    action positions the cable horizontally above the table and has the free arm grasp
    the other end of the cable. After execution both arms hold the cable.
    """

    cable_annotation: Cable
    """
    The cable semantic annotation to regrasp.
    """

    hanger_body: Body
    """
    Body of the cable hanger to hang the cable on.
    """

    regrasp_height: float = field(default=0.5)
    """
    Height in metres above the table surface where the cable center is positioned.
    """

    table_width: float = field(default=1.2)
    """
    Distance in metres along the hanger's front-facing axis from the world origin to the
    center of the cable.
    """

    table_depth: float = field(default=0.6)
    """
    Distance in metres along the hanger's front-facing axis from the world origin to the
    center of the cable.
    """

    approach_direction: int = 0
    """
    Index of the hanger's local axis that is the front-facing axis.

    0 is the X axis, 1 is the Y axis, 2 is the Z axis.
    """

    approach_sign: int = 1
    """
    Direction the approach axis is pointing.

    If the axis is pointing towards the approach direction the approach_sign is +1, if
    the axis is pointing to the back the approach_sign is -1.
    """

    pre_condition = staticmethod(_pre_condition_one_gripper_free)
    post_condition = staticmethod(_post_condition_cable_in_both_grippers)

    @property
    def _action_plan(self) -> PlanNode:
        holding_arm = _determine_holding_arm(self.cable_annotation.root, self.robot)
        free_arm = Arms.RIGHT if holding_arm == Arms.LEFT else Arms.LEFT
        side_sign = 1.0 if free_arm == Arms.LEFT else -1.0

        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )

        table_z = 0.605
        target_z = table_z + self.regrasp_height

        free_arm_end_effector = ViewManager.get_end_effector_view(free_arm, self.robot)

        holding_pose = self._build_mid_pose(
            up_offset=target_z,
            orientation=_gripper_orientation_from_z_axis(
                gripper_z_axis=-up_world,
                fallback_direction=np.array([0.0, 0.0, 1.0]),
                z_rotation=3 * pi / 2,
            ),
        )

        free_grasp_z = target_z - 0.01
        free_grasp_pose = self._build_mid_pose(
            up_offset=free_grasp_z,
            side_offset=0.04,
            orientation=_gripper_orientation_from_z_axis(
                gripper_z_axis=side_sign * side_world,
                fallback_direction=np.array([0.0, 0.0, 1.0]),
                z_rotation=pi,
            ).multiply(Quaternion.from_rpy(-pi / 4, 0.0, 0.0)),
        )

        spread_orientation = _gripper_orientation_from_z_axis(
            gripper_z_axis=self.approach_sign * front_world,
            fallback_direction=np.array([0.0, 0.0, 1.0]),
            z_rotation=pi / 2,
        )

        inter_holding_arm_pose = self._build_mid_pose(
            side_offset=-0.1,
            up_offset=target_z,
            orientation=spread_orientation,
        )

        inter_free_arm_pose = self._build_mid_pose(
            side_offset=0.1,
            up_offset=target_z - 0.1,
            orientation=spread_orientation,
        )

        half_length = 0.3
        hold_spread_pose = self._build_spread_pose(
            arm=holding_arm,
            z=target_z,
            half_length=half_length,
            orientation=spread_orientation,
        )
        free_spread_pose = self._build_spread_pose(
            arm=free_arm,
            z=target_z,
            half_length=half_length,
            orientation=spread_orientation,
        )

        print(f"Regrasping: holding={holding_arm.name}, free={free_arm.name}")
        print(f"Target holding pose: {holding_pose.to_position()}")
        print(f"Target free grasp pose: {free_grasp_pose.to_position()}")
        print(f"Target hold spread pose: {hold_spread_pose.to_position()}")
        print(f"Target free spread pose: {free_spread_pose.to_position()}")

        return sequential(
            children=[
                MoveGripperMotion(motion=GripperState.OPEN, gripper=free_arm),
                ParkArmsAction(holding_arm),
                MoveToolCenterPointMotion(
                    holding_pose,
                    holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    translate_pose_along_local_axis(
                        pose=free_grasp_pose, axis=[0, 0, 1], distance=-0.05
                    ),
                    free_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    free_grasp_pose,
                    free_arm,
                    movement_type=MovementType.CARTESIAN,
                    threshold=0.001,
                ),
                DAiSyFlexGripMotion(
                    motion=GripperState.FLEXCLOSE,
                    gripper=free_arm,
                    grip_position=0,
                    grip_force=20,
                ),
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=free_arm_end_effector.tool_frame,
                    parent_T_connection_expression=_attachment_transform(
                        free_arm_end_effector
                    ),
                ),
                MoveToolCenterPointMotion(
                    target=translate_pose_along_local_axis(
                        pose=free_grasp_pose, axis=[0, 1, 0], distance=0.3
                    ),
                    arm=free_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    target=translate_pose_along_local_axis(
                        pose=free_grasp_pose, axis=[0, 1, 0], distance=0.05
                    ),
                    arm=free_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    inter_holding_arm_pose,
                    holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    inter_free_arm_pose, free_arm, movement_type=MovementType.CARTESIAN
                ),
                MoveToolCenterPointMotion(
                    free_spread_pose,
                    free_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    hold_spread_pose,
                    holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveGripperMotion(motion=GripperState.OPEN, gripper=holding_arm),
            ],
        )

    def _build_mid_pose(
        self,
        orientation: Quaternion,
        front_offset: float = 0.0,
        side_offset: float = 0.0,
        up_offset: float = 0.0,
    ) -> Pose:
        """
        Build a pose at the center position between both arms along the hanger axes.

        :param up_offset: Height along the up axis in metres.
        :param orientation: The gripper orientation quaternion.
        """
        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )

        position = (
            front_world * self.table_depth / 2
            + side_world * (self.approach_sign * self.table_width / 2)
            + up_world * (up_offset - 0.0477)
            + front_world * front_offset
            + side_world * side_offset
        )

        return Pose(
            position=Point3(
                x=position[0],
                y=position[1],
                z=position[2],
                reference_frame=self.world.root,
            ),
            orientation=orientation,
            reference_frame=self.world.root,
        )

    def _build_spread_pose(
        self,
        arm: Arms,
        z: float,
        half_length: float,
        orientation: Quaternion,
    ) -> Pose:
        """
        Build a pose for spreading the arms horizontally along the side axis.

        The left arm moves along ``-side`` and the right arm along ``+side`` so the
        cable is stretched between them at the same height.

        :param arm: The arm to build the spread pose for.
        :param z: Height along the up axis in metres.
        :param half_length: Half the cable length in metres for the offset.
        :param orientation: The gripper orientation quaternion.
        """
        side_sign = -1.0 if arm == Arms.LEFT else 1.0

        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )

        position = (
            -front_world * self.approach_sign * self.table_depth / 2
            - side_world * (self.table_width / 2 - half_length * side_sign)
            + up_world * (z - 0.0477)
        )
        return Pose(
            position=Point3(
                x=position[0],
                y=position[1],
                z=position[2],
                reference_frame=self.world.root,
            ),
            orientation=orientation,
            reference_frame=self.world.root,
        )


# %% CableRehangAction


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

    approach_sign: int = 1
    """
    Direction the approach axis is pointing.

    If the axis is pointing towards the approach direction the approach_sign is +1, if
    the axis is pointing to the back the approach_sign is -1.
    """

    pre_condition = staticmethod(_pre_condition_one_gripper_free)
    post_condition = staticmethod(_post_condition_cable_in_either_gripper)

    @property
    def _action_plan(self) -> PlanNode:
        holding_arm = _determine_holding_arm(self.cable_annotation.root, self.robot)
        free_arm = Arms.RIGHT if holding_arm == Arms.LEFT else Arms.LEFT
        print(f"Holding with {holding_arm.name}, Free arm {free_arm.name}")

        hang_poses = self._calculate_hang_pose(holding_arm)

        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )

        inter_hang_pose = hang_poses["inter_hang_pose"]
        approach_hang_pose = hang_poses["approach_hang_pose"]
        pre_hang_pose = hang_poses["pre_hang_pose"]
        hang_pose = hang_poses["hang_pose"]

        print(f"Inter hang pose: {inter_hang_pose.to_position()}")
        print(f"Approach hang pose: {approach_hang_pose.to_position()}")
        print(f"Pre hang pose: {pre_hang_pose.to_position()}")
        print(f"Hang pose: {hang_pose.to_position()}")

        return sequential(
            [
                MoveToolCenterPointMotion(
                    target=inter_hang_pose,
                    arm=holding_arm,
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
                    threshold=0.001,
                ),
                MoveToolCenterPointMotion(
                    target=hang_pose,
                    arm=holding_arm,
                    movement_type=MovementType.STRAIGHT_TRANSLATION,
                    threshold=0.001,
                ),
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=self.hanger_body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=self.side_offset,
                        y=-self.front_offset,
                        z=-self.cable_annotation.length / 2,
                        reference_frame=self.hanger_body,
                    ),
                ),
                MoveGripperMotion(gripper=holding_arm, motion=GripperState.OPEN),
                MoveToolCenterPointMotion(
                    target=pre_hang_pose,
                    arm=holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
            ],
        )

    def _calculate_hang_pose(self, holding_arm: Arms) -> dict[str, Pose]:
        """
        Calculate the pose to hang the cable back on the hanger.
        """
        poses = {}

        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )
        side_sign = 1.0 if holding_arm == Arms.RIGHT else -1.0

        hang_pos = self._hanging_point_position().to_np()
        print(f"Hanging pose: {hang_pos[:3]}")

        inter_hang_orientation = _gripper_orientation_from_z_axis(
            gripper_z_axis=-up_world, fallback_direction=front_world, z_rotation=pi
        )
        hang_orientation = _gripper_orientation_from_z_axis(
            -front_world, side_world * side_sign, z_rotation=2 * pi
        )

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

        inter_hang_pos = approach_hang_pos[:3] - up_world * 0.2
        inter_hang_pose = Pose(
            position=Point3(
                x=inter_hang_pos[0],
                y=inter_hang_pos[1],
                z=inter_hang_pos[2],
                reference_frame=self.world.root,
            ),
            orientation=inter_hang_orientation,
            reference_frame=self.world.root,
        )

        poses["inter_hang_pose"] = inter_hang_pose

        return poses

    def _hanging_point_position(self) -> Point3:
        parent_global = self.hanger_body.global_transform
        front_world, side_world, up_world = _hanger_axes(
            self.cable_annotation.hanging_from.global_transform,
            self.approach_direction,
            self.approach_sign,
        )

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
