from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from coraplex.robot_plans.actions.core.cable_actions import CableGraspAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.cable import Cable

from ...plan_running import expand, run_plan


def test_cable_grasp_attaches_cable_to_end_effector(
    mutable_cable_hanger_world,
):
    world, view, extensions = mutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    assert len(cable_annotations) == 1
    cable_annotation = cable_annotations[0]

    run_plan(ParkArmsAction(arms=view.all_arms), extensions)
    run_plan(
        CableGraspAction(
            cable_annotation=cable_annotation,
            hanger_body=world.get_body_by_name(PrefixedName("hanger")),
            grasp_offset=0.1,
            front_offset=0.1,
        ),
        extensions,
    )

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

    assert attached_to_left or attached_to_right, (
        "Cable must be attached to one of the end effectors after grasp"
    )


@pytest.mark.parametrize(
    "target, approach_start, setback_of",
    [
        ("scoop_pose", "pre_scoop_pose", lambda action: action.scoop_setback),
        (
            "free_cable_pose",
            "pre_free_cable_pose",
            lambda action: action.free_cable_setback,
        ),
    ],
)
def test_scoop_arm_target_is_moved_back_against_its_approach_by_its_setback(
    immutable_cable_hanger_world,
    target: str,
    approach_start: str,
    setback_of: Callable[[CableGraspAction], float],
):
    """
    The scoop arm's target is moved back against its approach by the target's setback.
    """
    world, _, extensions = immutable_cable_hanger_world
    (cable_annotation,) = world.get_semantic_annotations_by_type(Cable)
    hanger_body = world.get_body_by_name(PrefixedName("hanger"))
    set_back = expand(
        CableGraspAction(cable_annotation=cable_annotation, hanger_body=hanger_body),
        extensions,
    )
    not_set_back = expand(
        CableGraspAction(
            cable_annotation=cable_annotation,
            hanger_body=hanger_body,
            scoop_setback=0.0,
            free_cable_setback=0.0,
        ),
        extensions,
    )
    scoop_arm = set_back._choose_scoop_arm()
    moved, original = (
        action._calculate_scoop_poses(scoop_arm, scoop_arm.end_effector)
        for action in (set_back, not_set_back)
    )
    displacement = (
        moved[target].position.to_np()[:3] - original[target].position.to_np()[:3]
    )
    approach = (
        original[target].position.to_np()[:3]
        - original[approach_start].position.to_np()[:3]
    )

    assert np.linalg.norm(displacement) == pytest.approx(setback_of(set_back))
    assert np.dot(displacement, approach) < 0


def test_only_the_scoop_arm_leaving_the_cable_held_by_both_grippers_is_not_replanned(
    immutable_cable_hanger_world,
):
    """
    Of the two moves of the scoop arm to the pre-free-cable pose, only the one with both
    grippers on the cable may not be replanned.
    """
    world, _, extensions = immutable_cable_hanger_world
    (cable_annotation,) = world.get_semantic_annotations_by_type(Cable)
    action = expand(
        CableGraspAction(
            cable_annotation=cable_annotation,
            hanger_body=world.get_body_by_name(PrefixedName("hanger")),
        ),
        extensions,
    )
    scoop_arm = action._choose_scoop_arm()
    pre_free_cable_position = action._calculate_scoop_poses(
        scoop_arm, scoop_arm.end_effector
    )["pre_free_cable_pose"].position.to_np()
    goals = [node for node in action.descendants if isinstance(node, CartesianPose)]
    to_pre_free_cable_pose = [
        goal
        for goal in goals
        if np.allclose(goal.goal_pose.position.to_np(), pre_free_cable_position)
    ]

    assert [goal.allow_replanning for goal in to_pre_free_cable_pose] == [True, False]
    assert all(
        goal.allow_replanning for goal in goals if goal not in to_pre_free_cable_pose
    )


def test_cable_grasp_chooses_closer_arm_for_scooping(
    immutable_cable_hanger_world,
):
    world, view, extensions = immutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    cable_annotation = cable_annotations[0]

    action = CableGraspAction(
        cable_annotation=cable_annotation,
        hanger_body=world.get_body_by_name(PrefixedName("hanger")),
        grasp_offset=0.1,
        front_offset=0.1,
    )
    run_plan(ParkArmsAction(arms=view.all_arms), extensions)
    run_plan(action, extensions)

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
