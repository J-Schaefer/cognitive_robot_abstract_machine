"""
Replanning stuck motions of the cable demo in this process, so the motion charts can run
on a controller elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from experiments.daisy_cable_replanning.collisions import ActiveMotionTracker
from experiments.daisy_cable_replanning.replanning import (
    ReplanningLog,
    ReplanningParameters,
)
from experiments.daisy_cable_replanning.stuck_motion_demo import StuckMotionCableDemo
from experiments.daisy_cable_replanning.stuck_motion_replanning import (
    PlannedPathAfterStuckMotion,
    PlannedToolCenterPointMotion,
    StuckMotionReplanning,
    ToolCenterPointMotionEndingWhenStuck,
)
from experiments.daisy_gripper_demo.demo import ExecutionChoice
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.utils import daisy_installed
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")

# %% helpers


def _transformation(tmp_path: Path) -> StuckMotionReplanning:
    return StuckMotionReplanning(
        parameters=ReplanningParameters(),
        replanning_log=ReplanningLog(
            path=tmp_path / "log.jsonl", trial=0, motion_tracker=ActiveMotionTracker()
        ),
    )


def _transformed_motion_nodes(world: World, target: Pose, tmp_path: Path) -> list:
    """
    :return: The children of a plan of one tool center point motion of the right arm to
        the target, after the transformation rewrote it.
    """
    robot = world.get_semantic_annotations_by_type(DAiSy)[0]
    plan_root = sequential(
        [MoveToolCenterPointMotion(target=target, arm=robot.right_arm)],
        Context(world, robot),
    )
    _transformation(tmp_path).apply(plan_root)
    return plan_root.children


# %% plan transformation


def test_tool_center_point_motion_is_followed_by_a_planned_path_to_its_target(
    parked_daisy_world: World, tmp_path: Path
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    target = Pose.from_xyz_rpy(
        x=0.1, reference_frame=robot.right_arm.end_effector.tool_frame
    )

    motion_node, path_node = _transformed_motion_nodes(
        parked_daisy_world, target, tmp_path
    )

    assert type(motion_node.designator) is ToolCenterPointMotionEndingWhenStuck
    assert type(path_node) is PlannedPathAfterStuckMotion
    assert path_node.stuck_motion is motion_node.designator
    (planned_motion_node,) = path_node.children
    assert type(planned_motion_node.designator) is PlannedToolCenterPointMotion
    assert (
        planned_motion_node.designator.target,
        planned_motion_node.designator.arm,
    ) == (
        target,
        robot.right_arm,
    )


# %% reaching the target


def test_motion_at_its_target_reached_it(parked_daisy_world: World, tmp_path: Path):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    target = robot.right_arm.end_effector.tool_frame.global_pose

    motion_node, _ = _transformed_motion_nodes(parked_daisy_world, target, tmp_path)

    assert motion_node.designator.reached_target


def test_motion_short_of_its_target_did_not_reach_it(
    parked_daisy_world: World, tmp_path: Path
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    target = Pose.from_xyz_rpy(
        x=0.1, reference_frame=robot.right_arm.end_effector.tool_frame
    )

    motion_node, _ = _transformed_motion_nodes(parked_daisy_world, target, tmp_path)

    assert not motion_node.designator.reached_target


# %% whole demo


def test_simulated_demo_completes_by_replanning_stuck_motions(tmp_path: Path):
    log = tmp_path / "replanning.jsonl"

    StuckMotionCableDemo(execution=ExecutionChoice.SIMULATED, replanning_log=log).run()

    assert log.read_text().count("\n") >= 1
