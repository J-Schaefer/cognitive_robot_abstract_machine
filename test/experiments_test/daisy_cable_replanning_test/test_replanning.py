"""
Recording the joint states in which DAiSy's arms collide, and planning paths around
them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from experiments.daisy_cable_replanning.cable_demo import DemoScript
from experiments.daisy_cable_replanning.collisions import (
    ActiveMotionTracker,
    ArmCollisionPairs,
    ArmCollisionRecorder,
    CollisionKind,
    IllegalJointStateDatabase,
)
from experiments.daisy_cable_replanning.replanning import (
    JointSpacePlanner,
    ProbedJointState,
    ReplanningParameters,
)
from experiments.daisy_cable_replanning.trials import (
    CableDemoTrial,
    TrialFiles,
    TrialNoise,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.utils import daisy_installed
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")

# %% helpers


def _robot(world: World) -> DAiSy:
    return world.get_semantic_annotations_by_type(DAiSy)[0]


def _record(
    world: World, log: Path, positions: np.ndarray
) -> IllegalJointStateDatabase:
    """
    :return: The database of what a recorder logged when the arms moved to the
        positions.
    """
    database = IllegalJointStateDatabase.for_daisy_arms()
    ArmCollisionRecorder(
        _world=world,
        robot=_robot(world),
        motion_tracker=ActiveMotionTracker(),
        log=log,
        trial=0,
        joint_names=database.joint_names,
    )
    connections = [world.get_connection_by_name(name) for name in database.joint_names]
    with world.batch_state_changes():
        for connection, position in zip(connections, positions):
            connection.position = position
    database.extend_from_log(log)
    return database


# %% recording


def test_recorder_logs_a_collision_within_the_right_arm(
    parked_daisy_world: World,
    right_forearm_touching_gripper: np.ndarray,
    tmp_path: Path,
):
    database = _record(
        parked_daisy_world, tmp_path / "log.jsonl", right_forearm_touching_gripper
    )

    (state,) = database.illegal_joint_states
    assert state.collision_kind == CollisionKind.WITHIN_ARM
    assert state.colliding_joint_names == [
        connection.name.name
        for connection in _robot(parked_daisy_world).right_arm.active_connections
    ]
    assert state.positions == pytest.approx(right_forearm_touching_gripper.tolist())


def test_recorder_logs_nothing_while_the_arms_are_clear(
    parked_daisy_world: World, tmp_path: Path
):
    parked = IllegalJointStateDatabase.for_daisy_arms().joint_configurations(
        parked_daisy_world
    )
    database = _record(
        parked_daisy_world, tmp_path / "log.jsonl", parked.current_positions()
    )

    assert database.illegal_joint_states == []


def test_database_survives_saving_and_loading(
    parked_daisy_world: World,
    right_forearm_touching_gripper: np.ndarray,
    tmp_path: Path,
):
    database = _record(
        parked_daisy_world, tmp_path / "log.jsonl", right_forearm_touching_gripper
    )
    database.save(tmp_path / "database.json")

    assert IllegalJointStateDatabase.load(tmp_path / "database.json") == database


def test_joint_configurations_compare_only_the_colliding_joints(
    parked_daisy_world: World,
    right_forearm_touching_gripper: np.ndarray,
    tmp_path: Path,
):
    database = _record(
        parked_daisy_world, tmp_path / "log.jsonl", right_forearm_touching_gripper
    )
    configurations = database.joint_configurations(parked_daisy_world)
    left_arm_moved_away = right_forearm_touching_gripper.copy()
    left_arm_moved_away[:6] += 1.0

    assert configurations.distance_to(left_arm_moved_away) == pytest.approx(0.0)


# %% planning


def _planner_around(world: World, illegal: np.ndarray) -> JointSpacePlanner:
    """
    :return: A planner for the right arm that keeps clear of the single illegal
        configuration of the right arm.
    """
    database = IllegalJointStateDatabase.for_daisy_arms()
    configurations = database.joint_configurations(world)
    configurations.positions = illegal.reshape(1, -1)
    configurations.relevant_connections = np.array([[False] * 6 + [True] * 6])
    robot = _robot(world)
    parameters = ReplanningParameters(planning_time=5.0)
    return JointSpacePlanner(
        world=world,
        collision_matrix=ArmCollisionPairs(
            world=world, robot=robot
        ).collision_matrix_of(robot.right_arm, parameters.collision_margin),
        illegal_configurations=configurations,
        parameters=parameters,
    )


def test_planned_path_keeps_clear_of_the_illegal_configuration(
    parked_daisy_world: World,
):
    connections = _robot(parked_daisy_world).right_arm.active_connections
    start = np.array([connection.position for connection in connections])
    goal = np.array(DemoScript.daisy_cable_demo().load().SafeArmPosture.RIGHT.value)
    illegal = np.concatenate([np.zeros(6), (start + goal) / 2])
    planner = _planner_around(parked_daisy_world, illegal)

    waypoints = planner.plan(connections, goal)

    assert waypoints[0] == pytest.approx(start)
    assert waypoints[-1] == pytest.approx(goal)
    assert all(
        np.max(np.abs(waypoint - illegal[6:])) >= planner.parameters.clearance
        for waypoint in waypoints
    )
    with ProbedJointState(world=parked_daisy_world, connections=connections) as probe:
        assert all(
            probe.is_collision_free(waypoint, planner.collision_matrix)
            for waypoint in waypoints
        )


def test_planning_leaves_the_world_state_unchanged(parked_daisy_world: World):
    connections = _robot(parked_daisy_world).right_arm.active_connections
    state_before = parked_daisy_world.state.positions.copy()
    goal = np.array(DemoScript.daisy_cable_demo().load().SafeArmPosture.RIGHT.value)
    planner = _planner_around(parked_daisy_world, np.full(12, 10.0))

    planner.plan(connections, goal)

    assert parked_daisy_world.state.positions == pytest.approx(state_before)


# %% whole demo


def test_replanning_avoids_the_collisions_a_recorded_trial_ran_into(tmp_path: Path):
    recording = CableDemoTrial(
        number=0, seed=5, files=TrialFiles(directory=tmp_path / "record", number=0)
    )
    recording.run()
    database = IllegalJointStateDatabase.for_daisy_arms()
    database.extend_from_log(recording.files.collisions)
    database.save(tmp_path / "illegal_joint_states.json")
    replanning = CableDemoTrial(
        number=0,
        seed=5,
        files=TrialFiles(directory=tmp_path / "replan", number=0),
        illegal_joint_states=tmp_path / "illegal_joint_states.json",
    )

    replanning.run()

    assert database.illegal_joint_states != []
    assert replanning.files.collisions.read_text() == ""
    assert replanning.files.replanning.read_text().count("\n") >= 1


# %% trial noise


def test_trial_noise_moves_only_the_post():
    layout = DemoScript.daisy_cable_demo().load().CableWorkcellLayout()
    noise = TrialNoise()

    perturbed = noise.perturbed_layout(layout, np.random.default_rng(0))

    assert perturbed.post_x != layout.post_x
    assert abs(perturbed.post_x - layout.post_x) <= noise.post_position
    assert abs(perturbed.post_y - layout.post_y) <= noise.post_position
    assert abs(perturbed.post_yaw - layout.post_yaw) <= noise.post_yaw
    assert replace(perturbed, post_x=0, post_y=0, post_yaw=0) == replace(
        layout, post_x=0, post_y=0, post_yaw=0
    )
