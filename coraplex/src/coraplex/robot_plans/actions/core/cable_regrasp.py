from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any, Dict

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, MovementType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperIsFree, GripperIsNotFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.cable_grasp import (
    _cross,
    _normalized,
    _rotation_matrix_from_axes,
)
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
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Quaternion,
)


@dataclass
class CableRegraspAction(ActionDescription):
    """
    Regrasps a cable that is already held by one arm.

    After the initial :class:`CableGraspAction`, one arm holds the cable and the other
    arm is free. This action positions the cable horizontally above the table and has
    the free arm grasp the other end of the cable. After execution both arms hold the
    cable.
    """

    cable_annotation: Cable
    """
    The cable semantic annotation to regrasp.
    """

    regrasp_height: float = field(default=0.15)
    """
    Height in metres above the table surface where the cable centre is positioned.
    """

    cable_centre_x: float = field(default=-0.32)
    """
    X coordinate in the world frame for the centre of the cable when horizontal.
    """

    @property
    def _action_plan(self) -> PlanNode:
        holding_arm = self._determine_holding_arm()
        free_arm = Arms.RIGHT if holding_arm == Arms.LEFT else Arms.LEFT

        holding_end_effector = ViewManager.get_end_effector_view(
            holding_arm, self.robot
        )

        cable_body = self.cable_annotation.root
        tool_frame = holding_end_effector.tool_frame

        tool_pose = tool_frame.global_pose
        cable_pose = cable_body.global_pose

        tool_T_cable = (
            tool_pose.to_homogeneous_matrix().inverse()
            @ cable_pose.to_homogeneous_matrix()
        )
        cable_T_tool = tool_T_cable.inverse()

        half_length = self.cable_annotation.length / 2.0

        cable_top_local = HomogeneousTransformationMatrix.from_xyz_rpy(
            0, 0, half_length, reference_frame=cable_body
        )
        cable_bottom_local = HomogeneousTransformationMatrix.from_xyz_rpy(
            0, 0, -half_length, reference_frame=cable_body
        )

        table_z = 0.605
        target_centre_np = np.array(
            [self.cable_centre_x, 0.0, table_z + self.regrasp_height]
        )

        target_cable_pose = Pose(
            position=Point3(
                x=target_centre_np[0],
                y=target_centre_np[1],
                z=target_centre_np[2],
                reference_frame=self.world.root,
            ),
            orientation=Quaternion.from_rpy(roll=-pi / 2, pitch=0, yaw=0),
            reference_frame=self.world.root,
        )

        target_cable_T = target_cable_pose.to_homogeneous_matrix()
        target_tool_T = target_cable_T @ cable_T_tool
        target_tool_pose = target_tool_T.to_pose()

        target_cable_after: Pose = (
            target_tool_pose.to_homogeneous_matrix() @ tool_T_cable
        ).to_pose()

        cable_top_after = (
            target_cable_after.to_homogeneous_matrix() @ cable_top_local.to_pose()
        )
        cable_bottom_after = (
            target_cable_after.to_homogeneous_matrix() @ cable_bottom_local.to_pose()
        )

        tool_after_pos = target_tool_pose.to_position().to_np()[:3]
        top_after_pos = cable_top_after.to_position().to_np()[:3]
        bottom_after_pos = cable_bottom_after.to_position().to_np()[:3]

        dist_to_top = np.linalg.norm(tool_after_pos - top_after_pos)
        dist_to_bottom = np.linalg.norm(tool_after_pos - bottom_after_pos)

        free_end_target = (
            cable_top_after if dist_to_top >= dist_to_bottom else cable_bottom_after
        )

        y_direction = np.array([0.0, 1.0, 0.0])
        side_sign = -1.0 if free_arm == Arms.RIGHT else 1.0
        side_direction = y_direction * side_sign

        free_grasp_orientation = self._grasp_gripper_orientation(
            side_direction, np.array([0.0, 0.0, -1.0])
        )

        free_end_pos = free_end_target.to_position()
        free_target_pose = Pose(
            position=Point3(
                x=free_end_pos.x,
                y=free_end_pos.y,
                z=free_end_pos.z,
                reference_frame=self.world.root,
            ),
            orientation=free_grasp_orientation,
            reference_frame=self.world.root,
        )

        print(f"Regrasping: holding={holding_arm.name}, free={free_arm.name}")
        print(f"Target holding pose: {target_tool_pose.to_position()}")
        print(f"Target free pose: {free_target_pose.to_position()}")

        return sequential(
            children=[
                MoveGripperMotion(motion=GripperState.OPEN, gripper=free_arm),
                MoveToolCenterPointMotion(
                    target_tool_pose,
                    holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    free_target_pose,
                    free_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveGripperMotion(motion=GripperState.CLOSE, gripper=free_arm),
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

    def _grasp_gripper_orientation(
        self,
        side_direction: np.ndarray,
        front_direction: np.ndarray,
    ) -> Quaternion:
        gripper_z = _normalized(side_direction)
        world_up = np.array([0, 0, 1])

        cross_xz = _cross(world_up, gripper_z)
        if np.linalg.norm(cross_xz) < 1e-6:
            fallback = _cross(world_up, front_direction)
            if np.linalg.norm(fallback) < 1e-6:
                gripper_x = np.array([1.0, 0.0, 0.0])
            else:
                gripper_x = _normalized(fallback)
        else:
            gripper_x = _normalized(cross_xz)
        gripper_y = _normalized(_cross(gripper_z, gripper_x))

        rotation_matrix = _rotation_matrix_from_axes(gripper_x, gripper_y, gripper_z)
        return Quaternion.from_rotation_matrix(rotation_matrix)

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
        return and_(
            is_body_in_gripper(variable_from(cable_body), left_end_effector) > 0.9,
            is_body_in_gripper(variable_from(cable_body), right_end_effector) > 0.9,
        )
