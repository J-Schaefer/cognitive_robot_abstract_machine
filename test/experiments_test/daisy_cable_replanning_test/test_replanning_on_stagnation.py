"""
Replanning the motions of the cable demo that collision avoidance leaves stuck.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from coraplex.plans.failures import MotionMadeNoProgress
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from experiments.daisy_cable_replanning.collisions import (
    ActiveMotionTracker,
    ArmCollisionPairs,
)
from experiments.daisy_cable_replanning.replanning import (
    ReplanningLog,
    ReplanningParameters,
    StagnationReplannedToolCenterPointMotion,
    StagnationReplanning,
)
from experiments.daisy_cable_replanning.trials import CableDemoTrial, TrialFiles
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.utils import daisy_installed
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")

# %% collision pairs


def test_arm_pairs_ignore_temporary_rules_allowing_the_arm_to_collide(
    parked_daisy_world: World,
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    pairs = ArmCollisionPairs(world=parked_daisy_world, robot=robot)
    checked_before = pairs.collision_matrix_of(robot.right_arm, 0.0).collision_checks
    parked_daisy_world.collision_manager.add_temporary_rule(
        AllowCollisionForBodies(
            allowed_collision_bodies=pairs.bodies_of_arm[robot.right_arm]
        )
    )

    checked_after = pairs.collision_matrix_of(robot.right_arm, 0.0).collision_checks

    assert checked_before != set()
    assert checked_after == checked_before


# %% plan transformation


def test_stagnation_replanning_keeps_the_parameters_of_the_motion(
    parked_daisy_world: World, tmp_path: Path
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    motion = MoveToolCenterPointMotion(
        target=Pose(reference_frame=parked_daisy_world.root), arm=robot.right_arm
    )
    transformation = StagnationReplanning(
        parameters=ReplanningParameters(),
        replanning_log=ReplanningLog(
            path=tmp_path / "log.jsonl", trial=0, motion_tracker=ActiveMotionTracker()
        ),
    )

    replacement = transformation.replacement_for(motion)

    assert type(replacement) is StagnationReplannedToolCenterPointMotion
    assert (replacement.target, replacement.arm) == (motion.target, motion.arm)
    assert replacement.parameters == transformation.parameters


# %% whole demo


def _trial(tmp_path: Path, name: str, replans_on_stagnation: bool) -> CableDemoTrial:
    return CableDemoTrial(
        number=0,
        seed=0,
        files=TrialFiles(directory=tmp_path / name, number=0),
        collision_avoidance=True,
        replans_on_stagnation=replans_on_stagnation,
    )


def test_collision_avoidance_alone_gets_stuck(tmp_path: Path):
    with pytest.raises(MotionMadeNoProgress):
        _trial(tmp_path, "avoidance", replans_on_stagnation=False).run()


def test_replanning_on_stagnation_completes_the_demo_without_collision(
    tmp_path: Path,
):
    trial = _trial(tmp_path, "replanning", replans_on_stagnation=True)

    trial.run()

    assert trial.files.collisions.read_text() == ""
    assert trial.files.replanning.read_text().count("\n") >= 1
