"""
Replanning the motions of the cable demo that collision avoidance leaves stuck.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from coraplex.plans.failures import MotionMadeNoProgress
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from experiments.daisy_cable_replanning.collisions import (
    ActiveMotionTracker,
    ArmCollisionPairs,
)
from experiments.daisy_cable_replanning.replanning import (
    PathSinceLastClearJointState,
    PathSinceMotionStart,
    ProbedJointState,
    ReplanningLog,
    ReplanningParameters,
    RetractAlongRememberedPath,
    RetractedStagnationReplannedToolCenterPointMotion,
    RetractTarget,
    StagnationReplannedToolCenterPointMotion,
    StagnationReplanning,
    ToolCenterPointMotionWithReplanning,
)
from experiments.daisy_cable_replanning.trials import CableDemoTrial, TrialFiles
from giskardpy.motion_statechart.context import MotionStatechartContext
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
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


# %% path since the last clear joint state


def _clearance_tracker(world: World, clearance: float) -> PathSinceLastClearJointState:
    """
    :return: A tracker of the right arm that counts a joint state as clear while every
        pair of the arm's bodies keeps the clearance.
    """
    robot = world.get_semantic_annotations_by_type(DAiSy)[0]
    pairs = ArmCollisionPairs(world=world, robot=robot)
    return PathSinceLastClearJointState(
        name="path since last clear joint state",
        connections=robot.right_arm.active_connections,
        clearances=CollisionMatrix(
            {
                CollisionCheck(check.body_a, check.body_b, clearance)
                for check in pairs.collision_matrix_of(
                    robot.right_arm, clearance
                ).collision_checks
            }
        ),
        waypoint_spacing=ReplanningParameters().waypoint_spacing,
    )


def _move_right_arm(
    world: World, tracker: PathSinceMotionStart, step: float
) -> np.ndarray:
    """
    :return: The positions the right arm was moved to, every joint the step further.
    """
    positions = np.array([connection.position for connection in tracker.connections])
    moved = positions + step
    ProbedJointState(world=world, connections=tracker.connections).write(moved)
    return moved


def test_tracker_restarts_the_path_while_the_arm_keeps_the_clearance(
    parked_daisy_world: World,
):
    tracker = _clearance_tracker(parked_daisy_world, clearance=0.0)
    context = MotionStatechartContext(world=parked_daisy_world)
    tracker.on_start(context)

    moved = _move_right_arm(
        parked_daisy_world, tracker, step=2 * tracker.waypoint_spacing
    )
    tracker.on_tick(context)

    assert tracker.path == [pytest.approx(moved)]


def test_tracker_extends_the_path_while_the_arm_does_not_keep_the_clearance(
    parked_daisy_world: World,
):
    tracker = _clearance_tracker(parked_daisy_world, clearance=10.0)
    context = MotionStatechartContext(world=parked_daisy_world)
    start = np.array([connection.position for connection in tracker.connections])
    tracker.on_start(context)

    _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing / 2)
    tracker.on_tick(context)
    long_step = _move_right_arm(
        parked_daisy_world, tracker, step=tracker.waypoint_spacing
    )
    tracker.on_tick(context)

    assert tracker.path == [pytest.approx(start), pytest.approx(long_step)]


def test_retract_follows_the_path_back_to_the_last_clear_joint_state(
    parked_daisy_world: World,
):
    tracker = _clearance_tracker(parked_daisy_world, clearance=10.0)
    context = MotionStatechartContext(world=parked_daisy_world)
    tracker.on_start(context)
    for _ in range(3):
        _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing)
        tracker.on_tick(context)
    retract = RetractAlongRememberedPath(name="retract", monitor=tracker)

    waypoints = retract.plan_waypoints(context)

    assert retract.connections == tracker.connections
    assert len(waypoints) == 4
    assert waypoints == list(reversed(tracker.path))


def test_path_since_motion_start_keeps_the_whole_path_while_the_arm_is_clear(
    parked_daisy_world: World,
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    tracker = PathSinceMotionStart(
        name="path since motion start",
        connections=robot.right_arm.active_connections,
        waypoint_spacing=ReplanningParameters().waypoint_spacing,
    )
    context = MotionStatechartContext(world=parked_daisy_world)
    start = np.array([connection.position for connection in tracker.connections])
    tracker.on_start(context)

    first = _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing)
    tracker.on_tick(context)
    second = _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing)
    tracker.on_tick(context)

    assert tracker.path == [
        pytest.approx(start),
        pytest.approx(first),
        pytest.approx(second),
    ]


# %% plan transformation


@pytest.mark.parametrize(
    "retract_target, replacement_type",
    [
        (None, StagnationReplannedToolCenterPointMotion),
        (
            RetractTarget.LAST_CLEAR_STATE,
            RetractedStagnationReplannedToolCenterPointMotion,
        ),
        (RetractTarget.MOTION_START, RetractedStagnationReplannedToolCenterPointMotion),
        (RetractTarget.RECENT_STEPS, RetractedStagnationReplannedToolCenterPointMotion),
    ],
)
def test_stagnation_replanning_keeps_the_parameters_of_the_motion(
    parked_daisy_world: World,
    tmp_path: Path,
    retract_target: RetractTarget | None,
    replacement_type: type[ToolCenterPointMotionWithReplanning],
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
        retract_target=retract_target,
    )

    replacement = transformation.replacement_for(motion)

    assert type(replacement) is replacement_type
    assert (replacement.target, replacement.arm) == (motion.target, motion.arm)
    assert replacement.parameters == transformation.parameters


# %% whole demo


def _trial(
    tmp_path: Path,
    name: str,
    replans_on_stagnation: bool,
    retract_target: RetractTarget | None = None,
) -> CableDemoTrial:
    return CableDemoTrial(
        number=0,
        seed=0,
        files=TrialFiles(directory=tmp_path / name, number=0),
        collision_avoidance=True,
        replans_on_stagnation=replans_on_stagnation,
        retract_target=retract_target,
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


def test_retracting_before_replanning_completes_the_demo_without_collision(
    tmp_path: Path,
):
    trial = _trial(
        tmp_path,
        "retracting",
        replans_on_stagnation=True,
        retract_target=RetractTarget.LAST_CLEAR_STATE,
    )

    trial.run()

    assert trial.files.collisions.read_text() == ""
    assert trial.files.replanning.read_text().count("\n") >= 1


def test_retracing_recent_steps_before_replanning_completes_the_demo_without_collision(
    tmp_path: Path,
):
    trial = _trial(
        tmp_path,
        "retracing",
        replans_on_stagnation=True,
        retract_target=RetractTarget.RECENT_STEPS,
    )

    trial.run()

    assert trial.files.collisions.read_text() == ""
    assert trial.files.replanning.read_text().count("\n") >= 1
