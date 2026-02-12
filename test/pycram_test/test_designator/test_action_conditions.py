import pytest

from krrood.entity_query_language.entity import (
    exists,
    get_false_statements,
    evaluate_condition,
)
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.failures import ConditionNotSatisfied
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import PickUpAction
from semantic_digital_twin.reasoning.predicates import reachable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body


def test_get_bound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )

    bound_variables = pick_action.get_bound_variables()

    assert len(bound_variables) == 3
    assert list(bound_variables.keys()) == [
        "object_designator",
        "arm",
        "grasp_description",
    ]
    assert list(bound_variables["arm"]._domain_) == [Arms.LEFT]
    assert bound_variables["arm"]._type_ == Arms
    assert list(bound_variables["object_designator"]._domain_) == [
        world.get_body_by_name("milk.stl")
    ]
    assert bound_variables["object_designator"]._type_ == Body


def test_get_unbound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )

    unbound_variables = pick_action.get_unbound_variables()

    assert len(unbound_variables) == 3


def test_pick_up_conditions(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    pre_condition = pick_action.pre_condition()
    post_condition = pick_action.post_condition()

    assert evaluate_condition(pre_condition) == False

    false_statements = get_false_statements(pre_condition)

    assert len(false_statements) == 1
    assert false_statements[0]._name_ == "reachable"

    with pytest.raises(ConditionNotSatisfied):
        pick_action.evaluate_pre_condition()

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.8, 2, 0
    )

    assert evaluate_condition(pre_condition) == True

    with simulated_robot:
        plan.perform()

    assert evaluate_condition(pre_condition) == False
    assert evaluate_condition(post_condition) == True
    assert pick_action.evaluate_post_condition() == True
