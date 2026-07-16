from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
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
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3

logger = logging.getLogger(__name__)


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

    approach_offset: float = field(default=0.1)
    """
    Distance in metres to approach the hanging point along the world Y axis.
    """

    @property
    def _action_plan(self) -> PlanNode:
        scoop_arm = self._choose_scoop_arm()
        grasp_arm = Arms.RIGHT if scoop_arm == Arms.LEFT else Arms.LEFT

        scoop_end_effector = ViewManager.get_end_effector_view(scoop_arm, self.robot)
        grasp_end_effector = ViewManager.get_end_effector_view(grasp_arm, self.robot)

        hanging_point = self.cable_annotation.hanging_from
        hanger_position = hanging_point.global_transform.to_position()

        scoop_position = Point3(
            x=hanger_position.x,
            y=hanger_position.y - self.approach_offset,
            z=hanger_position.z,
            reference_frame=self.world.root,
        )
        scoop_pose = Pose(
            position=scoop_position,
            orientation=scoop_end_effector.front_facing_orientation,
            reference_frame=self.world.root,
        )

        grasp_position = Point3(
            x=hanger_position.x,
            y=hanger_position.y - self.approach_offset,
            z=hanger_position.z - self.grasp_offset,
            reference_frame=self.world.root,
        )
        grasp_pose = Pose(
            position=grasp_position,
            orientation=grasp_end_effector.front_facing_orientation,
            reference_frame=self.world.root,
        )

        return sequential(
            children=[
                MoveGripperMotion(motion=GripperState.OPEN, gripper=scoop_arm),
                MoveGripperMotion(motion=GripperState.OPEN, gripper=grasp_arm),
                MoveToolCenterPointMotion(
                    scoop_pose,
                    scoop_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveToolCenterPointMotion(
                    grasp_pose,
                    grasp_arm,
                    movement_type=MovementType.CARTESIAN,
                ),
                MoveGripperMotion(motion=GripperState.CLOSE, gripper=grasp_arm),
                AttachNode(
                    body=self.cable_annotation.root,
                    new_parent=grasp_end_effector.tool_frame,
                ),
            ],
        )

    def _choose_scoop_arm(self) -> Arms:
        left_arm = ViewManager.get_arm_view(Arms.LEFT, self.robot)
        right_arm = ViewManager.get_arm_view(Arms.RIGHT, self.robot)

        hanging_point = self.cable_annotation.hanging_from
        hanger_pos = hanging_point.global_transform.to_position().to_np()

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
        """
        Both grippers must be free.
        """
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
        """
        The cable must be attached to one of the end effectors.
        """
        left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, context.robot)
        right_end_effector = ViewManager.get_end_effector_view(
            Arms.RIGHT, context.robot
        )
        cable_body = kwargs["cable_annotation"].root
        return or_(
            is_body_in_gripper(variable_from(cable_body), left_end_effector) > 0.9,
            is_body_in_gripper(variable_from(cable_body), right_end_effector) > 0.9,
        )
