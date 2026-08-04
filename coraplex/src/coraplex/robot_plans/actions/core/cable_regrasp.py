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
    _gripper_orientation_from_z_axis,
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

    regrasp_height: float = field(default=0.5)
    """
    Height in metres above the table surface where the cable centre is positioned.
    """

    cable_centre_x: float = field(default=-0.32)
    """
    Distance in metres along the hanger's front-facing axis from the world origin to the
    centre of the cable.
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

    @property
    def _action_plan(self) -> PlanNode:
        holding_arm = self._determine_holding_arm()
        free_arm = Arms.RIGHT if holding_arm == Arms.LEFT else Arms.LEFT

        front, side, up = self._hanger_axes()

        gripper_orientation = _gripper_orientation_from_z_axis(
            gripper_z_axis=front,
            fallback_direction=np.array([0.0, 0.0, 1.0]),
            z_rotation=pi,
        )

        table_z = 0.605
        target_z = table_z + self.regrasp_height

        # Moves the holding arm to the centre of the table between both
        # arms, raised above the table surface.
        holding_pose = self._build_mid_pose(
            target_z, front, up, gripper_orientation
        )

        # Positions the free arm beneath the holding arm so it can grasp
        # the cable from below.
        free_grasp_z = target_z - 0.08
        free_grasp_pose = self._build_mid_pose(
            free_grasp_z, front, up, gripper_orientation
        )

        # Spread both arms horizontally so the cable is stretched between
        # them. Left arm moves left, right arm moves right, both at the
        # same height.
        half_length = self.cable_annotation.length / 2.0
        hold_spread_pose = self._build_spread_pose(
            holding_arm,
            target_z,
            half_length,
            front,
            side,
            up,
            gripper_orientation,
        )
        free_spread_pose = self._build_spread_pose(
            free_arm,
            target_z,
            half_length,
            front,
            side,
            up,
            gripper_orientation,
        )

        print(f"Regrasping: holding={holding_arm.name}, free={free_arm.name}")
        print(f"Target holding pose: {holding_pose.to_position()}")
        print(f"Target free grasp pose: {free_grasp_pose.to_position()}")

        return sequential(
            children=[
                # Open the free arm's gripper before approaching.
                MoveGripperMotion(motion=GripperState.OPEN, gripper=free_arm),
                # Move holding arm to centre above table, side-up orientation.
                MoveToolCenterPointMotion(
                    holding_pose,
                    holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Move free arm beneath holding arm to grasp the lower
                # portion of the cable.
                MoveToolCenterPointMotion(
                    free_grasp_pose,
                    free_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                # Close gripper to capture the cable.
                MoveGripperMotion(motion=GripperState.CLOSE, gripper=free_arm),
                # Spread arms horizontally so the cable is held taut
                # between both grippers at the same height.
                MoveToolCenterPointMotion(
                    hold_spread_pose,
                    holding_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    free_spread_pose,
                    free_arm,
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

    def _hanger_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the hanger's (front, side, up) unit vectors in the world frame.

        Uses ``approach_direction`` and ``approach_sign`` to extract the correct axis
        from the hanger's global rotation, matching
        :meth:`CableGraspAction._hanger_axes`.
        """
        hanger_rot = self.cable_annotation.hanging_from.global_transform
        rot_np = np.array(hanger_rot.to_np()[:3, :3], dtype=float)
        front = self.approach_sign * rot_np[:, self.approach_direction]
        up = rot_np[:, 2]
        side = np.cross(up, front)
        return front, side, up

    def _build_mid_pose(
        self,
        z: float,
        front: np.ndarray,
        up: np.ndarray,
        orientation: Quaternion,
    ) -> Pose:
        """
        Build a pose at the centre position between both arms along the hanger axes.

        :param z: Height along the up axis in metres.
        :param front: World-frame unit vector for the hanger's front axis.
        :param up: World-frame unit vector for the hanger's up axis.
        :param orientation: The gripper orientation quaternion.
        """
        position = front * self.cable_centre_x + up * z
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
        front: np.ndarray,
        side: np.ndarray,
        up: np.ndarray,
        orientation: Quaternion,
    ) -> Pose:
        """
        Build a pose for spreading the arms horizontally along the side axis.

        The left arm moves along ``-side`` and the right arm along ``+side`` so the
        cable is stretched between them at the same height.

        :param arm: The arm to build the spread pose for.
        :param z: Height along the up axis in metres.
        :param half_length: Half the cable length in metres for the offset.
        :param front: World-frame unit vector for the hanger's front axis.
        :param side: World-frame unit vector for the hanger's side axis.
        :param up: World-frame unit vector for the hanger's up axis.
        :param orientation: The gripper orientation quaternion.
        """
        side_sign = -1.0 if arm == Arms.LEFT else 1.0
        position = front * self.cable_centre_x + side * (
            half_length * side_sign
        ) + up * z
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

    def _grasp_gripper_orientation(
        self,
        side_direction: np.ndarray,
        front_direction: np.ndarray,
    ) -> Quaternion:
        return _gripper_orientation_from_z_axis(side_direction, front_direction)

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
