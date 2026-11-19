"""
Replanning the motions of the cable demo that collision avoidance leaves stuck.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.plans.failures import MotionMadeNoProgress
from coraplex.plans.plan_transformation import PlanRewriting
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.failure_handling import HandledOnFailure
from cramph.node import StatechartNode
from cramph.statechart import Statechart
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
    RetractTarget,
    StagnationReplanning,
)
from experiments.daisy_cable_replanning.trials import CableDemoTrial, TrialFiles
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(
    not DAiSy.is_description_installed(), reason="DAiSy not installed"
)

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


def _run_in_a_statechart(world: World, node: StatechartNode) -> None:
    """
    Add the node to a statechart of the world and start it, as the statechart does when
    the node starts running.
    """
    Statechart(context=StatechartContext(world=world)).add_node(node=node)
    node.on_start()


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
    _run_in_a_statechart(parked_daisy_world, tracker)

    moved = _move_right_arm(
        parked_daisy_world, tracker, step=2 * tracker.waypoint_spacing
    )
    tracker.on_tick()

    assert tracker.passed_positions == [pytest.approx(moved)]


def test_tracker_extends_the_path_while_the_arm_does_not_keep_the_clearance(
    parked_daisy_world: World,
):
    tracker = _clearance_tracker(parked_daisy_world, clearance=10.0)
    start = np.array([connection.position for connection in tracker.connections])
    _run_in_a_statechart(parked_daisy_world, tracker)

    _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing / 2)
    tracker.on_tick()
    long_step = _move_right_arm(
        parked_daisy_world, tracker, step=tracker.waypoint_spacing
    )
    tracker.on_tick()

    assert tracker.passed_positions == [pytest.approx(start), pytest.approx(long_step)]


def test_retract_follows_the_path_back_to_the_last_clear_joint_state(
    parked_daisy_world: World,
):
    tracker = _clearance_tracker(parked_daisy_world, clearance=10.0)
    _run_in_a_statechart(parked_daisy_world, tracker)
    for _ in range(3):
        _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing)
        tracker.on_tick()
    retract = RetractAlongRememberedPath(name="retract", monitor=tracker)

    waypoints = retract.plan_waypoints()

    assert retract.connections == tracker.connections
    assert len(waypoints) == 4
    assert waypoints == list(reversed(tracker.passed_positions))


def test_path_since_motion_start_keeps_the_whole_path_while_the_arm_is_clear(
    parked_daisy_world: World,
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    tracker = PathSinceMotionStart(
        name="path since motion start",
        connections=robot.right_arm.active_connections,
        waypoint_spacing=ReplanningParameters().waypoint_spacing,
    )
    start = np.array([connection.position for connection in tracker.connections])
    _run_in_a_statechart(parked_daisy_world, tracker)

    first = _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing)
    tracker.on_tick()
    second = _move_right_arm(parked_daisy_world, tracker, step=tracker.waypoint_spacing)
    tracker.on_tick()

    assert tracker.passed_positions == [
        pytest.approx(start),
        pytest.approx(first),
        pytest.approx(second),
    ]


# %% plan transformation


@pytest.mark.parametrize("retract_target", [None, *RetractTarget])
def test_stagnation_replanning_puts_a_copy_of_the_goal_under_failure_handling(
    parked_daisy_world: World,
    tmp_path: Path,
    retract_target: RetractTarget | None,
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    original = CartesianPose(
        tip_link=robot.right_arm.end_effector.tool_frame,
        goal_pose=Pose(reference_frame=parked_daisy_world.root),
    )
    plan = Sequence(nodes=[original])
    transformation = StagnationReplanning(
        parameters=ReplanningParameters(),
        replanning_log=ReplanningLog(
            path=tmp_path / "log.jsonl", trial=0, motion_tracker=ActiveMotionTracker()
        ),
        retract_target=retract_target,
    )
    executor = KinematicallySimulatedPlanExecutor(
        context=StatechartContext.with_extensions(
            world=parked_daisy_world,
            extensions=[
                RobotAccess(robot=robot),
                PlanRewriting(transformations=[transformation]),
            ],
        )
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=plan)

    executor.prepare(statechart)

    (handled,) = plan.nodes
    goals = [node for node in plan.descendants if isinstance(node, CartesianPose)]
    assert type(handled) is HandledOnFailure
    assert [goal in handled.descendants for goal in goals] == [True]
    assert [(goal.goal_pose, goal.tip_link) for goal in goals] == [
        (original.goal_pose, original.tip_link)
    ]


def test_a_goal_that_does_not_allow_replanning_is_left_as_it_is(
    parked_daisy_world: World, tmp_path: Path
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    transformation = StagnationReplanning(
        parameters=ReplanningParameters(),
        replanning_log=ReplanningLog(
            path=tmp_path / "log.jsonl", trial=0, motion_tracker=ActiveMotionTracker()
        ),
    )

    assert not transformation.is_applicable(
        CartesianPose(
            tip_link=robot.right_arm.end_effector.tool_frame,
            goal_pose=Pose(reference_frame=parked_daisy_world.root),
            allow_replanning=False,
        )
    )


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
