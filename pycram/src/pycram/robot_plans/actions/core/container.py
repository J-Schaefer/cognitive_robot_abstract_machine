from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from sympy import true

from krrood.entity_query_language.entity import entity, variable, and_, or_
from krrood.entity_query_language.entity_result_processors import an
from krrood.entity_query_language.symbolic import Variable, SymbolicExpression
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body, Connection
from typing_extensions import Union, Optional, Type, Any, Iterable, Dict

from .pick_up import GraspingActionDescription
from ...motions.container import OpeningMotion, ClosingMotion
from ...motions.gripper import MoveGripperMotion
from ....config.action_conf import ActionConfig
from ....datastructures.dataclasses import Context
from ....datastructures.enums import (
    Arms,
    ContainerManipulationType,
    ApproachDirection,
    VerticalAlignment,
)
from ....datastructures.grasp import GraspDescription
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped
from ....failures import ContainerManipulationError
from ....language import SequentialPlan
from ....pose_validator import reachability_validator
from ....querying.predicates import GripperIsFree
from ....view_manager import ViewManager
from ....robot_plans.actions.base import ActionDescription, DescriptionType


@dataclass
class OpenAction(ActionDescription):
    """
    Opens a container like object
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be opened
    """
    arm: Arms
    """
    Arm that should be used for opening the container
    """
    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters the gripper should be at in the x-axis away from the handle.
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot_view)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
        )

        SequentialPlan(
            self.context,
            GraspingActionDescription(
                self.object_designator, self.arm, grasp_description
            ),
            OpeningMotion(self.object_designator, self.arm),
            MoveGripperMotion(
                GripperState.OPEN, self.arm, allow_gripper_collision=True
            ),
        ).perform()

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)

        return and_(
            GripperIsFree(manipulator),
            reachability_validator(
                PoseStamped.from_spatial_type(kwargs["object_designator"].global_pose),
                manipulator.tool_frame,
                context.robot,
                context.world,
                context.robot.full_body_controlled,
            ),
        )

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        manipulator = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        return is_body_in_gripper(
            kwargs["object_designator"], manipulator
        ) > 0.9 or np.allclose(
            kwargs["object_designator"].global_pose.to_position(),
            ViewManager.get_end_effector_view(
                kwargs["arm"], context.robot
            ).tool_frame.global_pose.to_position(),
            atol=3e-2,
        )

    @classmethod
    def description(
        cls,
        object_designator_description: DescriptionType[Body],
        arm: DescriptionType[Arms] = None,
        grasping_prepose_distance: DescriptionType[
            float
        ] = ActionConfig.grasping_prepose_distance,
    ) -> PartialDesignator[OpenAction]:
        return PartialDesignator[OpenAction](
            OpenAction,
            object_designator=object_designator_description,
            arm=arm,
            grasping_prepose_distance=grasping_prepose_distance,
        )


@dataclass
class CloseAction(ActionDescription):
    """
    Closes a container like object.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be closed
    """
    arm: Arms
    """
    Arm that should be used for closing
    """
    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters between the gripper and the handle before approaching to grasp.
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot_view)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
        )

        SequentialPlan(
            self.context,
            GraspingActionDescription(
                self.object_designator, self.arm, grasp_description
            ),
            ClosingMotion(self.object_designator, self.arm),
            MoveGripperMotion(
                GripperState.OPEN, self.arm, allow_gripper_collision=True
            ),
        ).perform()

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        close_connection = kwargs[
            "object_designator"
        ].get_first_parent_connection_of_type(ActiveConnection1DOF)

        return bool(close_connection.position < 0.1)

    @classmethod
    def description(
        cls,
        object_designator_description: DescriptionType[Body],
        arm: DescriptionType[Arms] = None,
        grasping_prepose_distance: DescriptionType[
            float
        ] = ActionConfig.grasping_prepose_distance,
    ) -> PartialDesignator[CloseAction]:
        return PartialDesignator[CloseAction](
            CloseAction,
            object_designator=object_designator_description,
            arm=arm,
            grasping_prepose_distance=grasping_prepose_distance,
        )


OpenActionDescription = OpenAction.description
CloseActionDescription = CloseAction.description
