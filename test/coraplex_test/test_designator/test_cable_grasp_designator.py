from __future__ import annotations

import numpy as np
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.cable_actions import CableGraspAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.cable import Cable


def test_cable_grasp_attaches_cable_to_end_effector(
    mutable_cable_hanger_world,
):
    world, view, context = mutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    assert len(cable_annotations) == 1
    cable_annotation = cable_annotations[0]

    plan = sequential(
        [
            ParkArmsAction(view.all_arms),
            CableGraspAction(
                cable_annotation=cable_annotation,
                hanger_body=world.get_body_by_name(PrefixedName("hanger")),
                grasp_offset=0.1,
                front_offset=0.1,
            ),
        ],
        context=context,
    ).plan

    with kinematically_simulated_robot:
        plan.perform()

    left_arm = view.left_arm
    right_arm = view.right_arm

    cable_body = cable_annotation.root
    attached_to_left = (
        world.get_connection(
            left_arm.end_effector.tool_frame,
            cable_body,
        )
        is not None
    )
    attached_to_right = (
        world.get_connection(
            right_arm.end_effector.tool_frame,
            cable_body,
        )
        is not None
    )

    assert (
        attached_to_left or attached_to_right
    ), "Cable must be attached to one of the end effectors after grasp"


def test_cable_grasp_chooses_closer_arm_for_scooping(
    immutable_cable_hanger_world,
):
    world, view, context = immutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    cable_annotation = cable_annotations[0]

    action = CableGraspAction(
        cable_annotation=cable_annotation,
        hanger_body=world.get_body_by_name(PrefixedName("hanger")),
        grasp_offset=0.1,
        front_offset=0.1,
    )
    plan = sequential([ParkArmsAction(view.all_arms), action], context=context).plan
    with kinematically_simulated_robot:
        plan.perform()

    scoop_arm = action._choose_scoop_arm()
    grasp_arm = view.right_arm if scoop_arm == view.left_arm else view.left_arm

    left_arm = view.left_arm
    right_arm = view.right_arm

    hanger_pos = action._hanging_point_position().to_np()

    left_tip_pos = left_arm.end_effector.tool_frame.global_transform.position.to_np()
    right_tip_pos = right_arm.end_effector.tool_frame.global_transform.position.to_np()

    left_distance = float(np.linalg.norm(left_tip_pos - hanger_pos))
    right_distance = float(np.linalg.norm(right_tip_pos - hanger_pos))

    if left_distance <= right_distance:
        assert scoop_arm == view.left_arm
    else:
        assert scoop_arm == view.right_arm

    assert scoop_arm != grasp_arm
