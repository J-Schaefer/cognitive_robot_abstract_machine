"""
Replanning stuck motions of the cable demo in this process, so the statechart of a plan
can run on a controller elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.plans.plan_transformation import PlanRewriting
from cramph.composites import (
    ChildChooserAccess,
    Parallel,
    Sequence,
    StoppedWhenTrue,
    TryInOrder,
)
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.daisy_cable_replanning.collisions import ActiveMotionTracker
from experiments.daisy_cable_replanning.replanning import (
    ReplanningLog,
    ReplanningParameters,
)
from experiments.daisy_cable_replanning.stuck_motion_demo import StuckMotionCableDemo
from experiments.daisy_cable_replanning.stuck_motion_replanning import (
    JointPathPlanning,
    JointPathToPose,
    StuckMotionReplanning,
)
from experiments.daisy_gripper_demo.demo import ExecutionChoice
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import FixedJointWaypointPath
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(
    not DAiSy.is_description_installed(), reason="DAiSy not installed"
)

# %% helpers


def _replanning_log(tmp_path: Path) -> ReplanningLog:
    """
    :return: A log of the planned paths in the test's directory.
    """
    return ReplanningLog(
        path=tmp_path / "log.jsonl", trial=0, motion_tracker=ActiveMotionTracker()
    )


def _target_ahead_of_the_right_tool_frame(world: World) -> Pose:
    """
    :return: A pose 10 cm along the x axis of the right tool frame.
    """
    robot = world.get_semantic_annotations_by_type(DAiSy)[0]
    return Pose.from_xyz_rpy(
        x=0.1, reference_frame=robot.right_arm.end_effector.tool_frame
    )


def _prepared_plan(
    world: World, tmp_path: Path, goal: CartesianPose
) -> tuple[Sequence, KinematicallySimulatedPlanExecutor]:
    """
    :return: A sequence holding the goal, rewritten by the transformation, and the
        executor that prepared it.
    """
    robot = world.get_semantic_annotations_by_type(DAiSy)[0]
    executor = KinematicallySimulatedPlanExecutor(
        context=StatechartContext.with_extensions(
            world=world,
            extensions=[
                RobotAccess(robot=robot),
                PlanRewriting(
                    transformations=[
                        StuckMotionReplanning(
                            parameters=ReplanningParameters(),
                            replanning_log=_replanning_log(tmp_path),
                        )
                    ]
                ),
            ],
        )
    )
    plan = Sequence(nodes=[goal])
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=plan)
    executor.prepare(statechart)
    return plan, executor


def _right_arm_goal(world: World, allow_replanning: bool = True) -> CartesianPose:
    """
    :return: A pose goal of the right tool frame to a target ahead of it.
    """
    robot = world.get_semantic_annotations_by_type(DAiSy)[0]
    return CartesianPose(
        tip_link=robot.right_arm.end_effector.tool_frame,
        goal_pose=_target_ahead_of_the_right_tool_frame(world),
        allow_replanning=allow_replanning,
    )


# %% plan transformation


def test_stuck_goal_is_followed_by_a_path_planned_to_its_pose(
    parked_daisy_world: World, tmp_path: Path
):
    """
    The goal is stopped once it stalls, and a path to its pose takes its place.
    """
    original = _right_arm_goal(parked_daisy_world)

    plan, _ = _prepared_plan(parked_daisy_world, tmp_path, original)

    (replacement,) = plan.nodes
    stopped_goal, path = replacement.nodes
    assert type(replacement) is TryInOrder
    assert type(stopped_goal) is StoppedWhenTrue
    assert type(path) is JointPathToPose
    assert (path.tip_link, path.goal_pose) == (original.tip_link, original.goal_pose)


def test_goal_not_allowing_replanning_is_left_unchanged(
    parked_daisy_world: World, tmp_path: Path
):
    """
    A goal that must not be replanned stays the step of its sequence.
    """
    original = _right_arm_goal(parked_daisy_world, allow_replanning=False)

    plan, _ = _prepared_plan(parked_daisy_world, tmp_path, original)

    assert plan.nodes == [original]


# %% planning in this process


def test_path_planning_hands_a_path_from_where_the_arm_is(
    parked_daisy_world: World, tmp_path: Path
):
    """
    The chosen child follows waypoints from the arm's current joint state, and the path
    is logged.
    """
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    node = JointPathToPose(
        tip_link=robot.right_arm.end_effector.tool_frame,
        goal_pose=_target_ahead_of_the_right_tool_frame(parked_daisy_world),
    )
    planning = JointPathPlanning(replanning_log=_replanning_log(tmp_path))
    Statechart(
        context=StatechartContext.with_extensions(
            world=parked_daisy_world,
            extensions=[
                RobotAccess(robot=robot),
                ChildChooserAccess(choosers=[planning]),
            ],
        )
    ).add_node(node=node)
    start = np.array(
        [connection.position for connection in robot.right_arm.active_connections]
    )

    child = planning.choose_child(node=node)

    (path,) = [part for part in child.nodes if isinstance(part, FixedJointWaypointPath)]
    assert type(child) is Parallel
    assert path.connections == robot.right_arm.active_connections
    assert path.waypoints[0] == pytest.approx(start.tolist())
    assert planning.replanning_log.path.read_text().count("\n") == 1


# %% whole demo


def test_simulated_demo_completes_by_replanning_stuck_motions(tmp_path: Path):
    """
    The simulated demo plans at least one path in this process and completes.
    """
    log = tmp_path / "replanning.jsonl"

    StuckMotionCableDemo(execution=ExecutionChoice.SIMULATED, replanning_log=log).run()

    assert log.read_text().count("\n") >= 1
