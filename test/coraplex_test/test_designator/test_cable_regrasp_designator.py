from __future__ import annotations

from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.cable_actions import (
    CableGraspAction,
    CableRegraspAction,
    _determine_holding_arm,
)
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.semantic_annotations.cable import Cable


def test_regrasp_identifies_holding_arm(mutable_cable_hanger_world):
    world, view, context = mutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    assert len(cable_annotations) == 1
    cable_annotation = cable_annotations[0]

    grasp_action = CableGraspAction(
        cable_annotation=cable_annotation,
        hanger_body=world.get_body_by_name(PrefixedName("hanger")),
        grasp_offset=0.1,
        front_offset=0.1,
    )
    plan = sequential(
        [ParkArmsAction(arm=Arms.BOTH), grasp_action],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    regrasp_action = CableRegraspAction(
        cable_annotation=cable_annotation,
        hanger_body=world.get_body_by_name(PrefixedName("hanger")),
    )
    regrasp_plan = sequential(
        [regrasp_action],
        context=context,
    ).plan

    with simulated_robot:
        regrasp_plan.perform()

    holding_arm = _determine_holding_arm(cable_annotation.root, view)
    free_arm = Arms.RIGHT if holding_arm == Arms.LEFT else Arms.LEFT

    assert holding_arm != free_arm

    cable_body = cable_annotation.root
    parent = cable_body.parent_kinematic_structure_entity
    holding_end_effector = ViewManager.get_end_effector_view(holding_arm, view)
    assert parent == holding_end_effector.tool_frame


def test_regrasp_cable_held_by_both_arms(mutable_cable_hanger_world):
    world, view, context = mutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    assert len(cable_annotations) == 1
    cable_annotation = cable_annotations[0]

    plan = sequential(
        [
            ParkArmsAction(arm=Arms.BOTH),
            CableGraspAction(
                cable_annotation=cable_annotation,
                hanger_body=world.get_body_by_name(PrefixedName("hanger")),
                grasp_offset=0.1,
                front_offset=0.1,
            ),
            CableRegraspAction(
                cable_annotation=cable_annotation,
                hanger_body=world.get_body_by_name(PrefixedName("hanger")),
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    left_end_effector = ViewManager.get_end_effector_view(Arms.LEFT, view)
    right_end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, view)
    cable_body = cable_annotation.root

    left_result = is_body_in_gripper(cable_body, left_end_effector)
    right_result = is_body_in_gripper(cable_body, right_end_effector)

    assert left_result > 0.9, "Cable must be in the left gripper after regrasp"
    assert right_result > 0.9, "Cable must be in the right gripper after regrasp"
