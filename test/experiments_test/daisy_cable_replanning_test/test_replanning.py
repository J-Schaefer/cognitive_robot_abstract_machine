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
from experiments.daisy_cable_replanning.exceptions import NoValidGoalConfiguration
from experiments.daisy_cable_replanning.replanning import (
    GoalConfigurationSampler,
    JointSpacePlanner,
    ProbedJointState,
    ReplanningParameters,
)
from experiments.daisy_cable_replanning.trials import (
    CableDemoTrial,
    TrialFiles,
    TrialNoise,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types.spatial_types import Pose
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
        robot=robot,
        arm=robot.right_arm,
        ignored_bodies={
            cable.root for cable in world.get_semantic_annotations_by_type(Cable)
        },
        illegal_configurations=configurations,
        parameters=parameters,
    )


def test_planned_path_keeps_clear_of_the_illegal_configuration(
    parked_daisy_world: World, safe_right_posture: tuple[float, ...]
):
    connections = _robot(parked_daisy_world).right_arm.active_connections
    start = np.array([connection.position for connection in connections])
    goal = np.array(safe_right_posture)
    illegal = np.concatenate([np.zeros(6), (start + goal) / 2])
    planner = _planner_around(parked_daisy_world, illegal)

    waypoints = planner.plan([goal])

    assert waypoints[0] == pytest.approx(start)
    assert waypoints[-1] == pytest.approx(goal)
    assert all(
        np.max(np.abs(waypoint - illegal[6:])) >= planner.parameters.clearance
        for waypoint in waypoints
    )
    with ProbedJointState(world=parked_daisy_world, connections=connections) as probe:
        assert all(
            probe.is_collision_free(waypoint, planner.collision_matrix())
            for waypoint in waypoints
        )


def test_planning_leaves_the_world_state_unchanged(
    parked_daisy_world: World, safe_right_posture: tuple[float, ...]
):
    state_before = parked_daisy_world.state.positions.copy()
    goal = np.array(safe_right_posture)
    planner = _planner_around(parked_daisy_world, np.full(12, 10.0))

    planner.plan([goal])

    assert parked_daisy_world.state.positions == pytest.approx(state_before)


# %% whole demo


def test_replanning_avoids_the_collisions_a_recorded_trial_ran_into(tmp_path: Path):
    recording = CableDemoTrial(
        number=0, seed=8, files=TrialFiles(directory=tmp_path / "record", number=0)
    )
    recording.run()
    database = IllegalJointStateDatabase.for_daisy_arms()
    database.extend_from_log(recording.files.collisions)
    database.save(tmp_path / "illegal_joint_states.json")
    replanning = CableDemoTrial(
        number=0,
        seed=8,
        files=TrialFiles(directory=tmp_path / "replan", number=0),
        illegal_joint_states=[tmp_path / "illegal_joint_states.json"],
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


def test_planner_leaves_a_cable_unchecked(parked_daisy_world: World):
    robot = _robot(parked_daisy_world)
    gripper = robot.right_arm.end_effector
    with parked_daisy_world.modify_world():
        cable = Cable.create_with_new_body_in_world(
            name=PrefixedName("cable"),
            world=parked_daisy_world,
            hanging_from=gripper.tool_frame,
            length=0.1,
        ).root
    planner = _planner_around(parked_daisy_world, np.full(12, 10.0))
    arm_pairs = ArmCollisionPairs(world=parked_daisy_world, robot=robot)

    checked_bodies = {
        body
        for check in planner.collision_matrix().collision_checks
        for body in (check.body_a, check.body_b)
    }

    assert cable in {
        body
        for check in arm_pairs.collision_matrix_of(
            robot.right_arm, 0.0
        ).collision_checks
        for body in (check.body_a, check.body_b)
    }
    assert cable not in checked_bodies
    assert checked_bodies & arm_pairs.bodies_of_arm[robot.right_arm]


def test_probe_counts_only_pairs_closer_than_their_distance(parked_daisy_world: World):
    robot = _robot(parked_daisy_world)
    connections = robot.right_arm.active_connections
    pairs = ArmCollisionPairs(world=parked_daisy_world, robot=robot)
    far = pairs.collision_matrix_of(robot.right_arm, distance=1.0)
    closest = min(
        contact.distance
        for contact in parked_daisy_world.collision_manager.collision_detector.check_collisions(
            far
        ).contacts
    )
    positions = np.array([connection.position for connection in connections])

    with ProbedJointState(world=parked_daisy_world, connections=connections) as probe:
        free_below = probe.is_collision_free(
            positions, pairs.collision_matrix_of(robot.right_arm, closest * 0.9)
        )
        free_above = probe.is_collision_free(
            positions, pairs.collision_matrix_of(robot.right_arm, closest * 1.1)
        )

    assert closest > 0
    assert free_below
    assert not free_above


# %% goal configurations


def _tool_frame_pose_at(world: World, positions: np.ndarray) -> Pose:
    """
    :return: The pose of the right tool frame with the right arm at the positions.
    """
    robot = _robot(world)
    with ProbedJointState(
        world=world, connections=robot.right_arm.active_connections
    ) as probe:
        probe.write(positions)
        return robot.right_arm.end_effector.tool_frame.global_transform.pose


def test_goal_configurations_put_the_tool_frame_at_the_pose(
    parked_daisy_world: World, safe_right_posture: tuple[float, ...]
):
    robot = _robot(parked_daisy_world)
    pose = _tool_frame_pose_at(parked_daisy_world, np.array(safe_right_posture))
    state_before = parked_daisy_world.state.positions.copy()
    sampler = GoalConfigurationSampler(
        world=parked_daisy_world,
        arm=robot.right_arm,
        root=robot.root,
        parameters=ReplanningParameters(goal_seeds=4),
    )

    configurations = sampler.configurations_reaching(pose)

    assert configurations != []
    for configuration in configurations:
        reached = _tool_frame_pose_at(parked_daisy_world, configuration)
        assert reached.position.to_np() == pytest.approx(
            pose.position.to_np(), abs=0.01
        )
    assert parked_daisy_world.state.positions == pytest.approx(state_before)


def test_planner_reaches_the_valid_one_of_several_goals(
    parked_daisy_world: World, safe_right_posture: tuple[float, ...]
):
    goal = np.array(safe_right_posture)
    illegal_goal = goal + 0.5
    planner = _planner_around(
        parked_daisy_world, np.concatenate([np.zeros(6), illegal_goal])
    )

    waypoints = planner.plan([illegal_goal, goal])

    assert waypoints[-1] == pytest.approx(goal)


def test_planner_reports_a_goal_too_close_to_an_illegal_configuration(
    parked_daisy_world: World, safe_right_posture: tuple[float, ...]
):
    illegal_goal = np.array(safe_right_posture)
    planner = _planner_around(
        parked_daisy_world, np.concatenate([np.zeros(6), illegal_goal])
    )

    with pytest.raises(NoValidGoalConfiguration) as error:
        planner.plan([illegal_goal])

    (rejected,) = error.value.rejected_goals
    assert rejected.positions == pytest.approx(illegal_goal.tolist())
    assert rejected.distance_to_illegal_configurations == pytest.approx(0.0)
    assert rejected.contacts == []


def test_planner_reports_the_contacts_of_a_colliding_goal(
    parked_daisy_world: World, right_forearm_touching_gripper: np.ndarray
):
    colliding_goal = right_forearm_touching_gripper[6:]
    planner = _planner_around(parked_daisy_world, np.full(12, 10.0))
    connections = _robot(parked_daisy_world).right_arm.active_connections
    with ProbedJointState(world=parked_daisy_world, connections=connections) as probe:
        expected_contacts = probe.contacts_closer_than_allowed(
            colliding_goal, planner.collision_matrix()
        )

    with pytest.raises(NoValidGoalConfiguration) as error:
        planner.plan([colliding_goal])

    (rejected,) = error.value.rejected_goals
    assert expected_contacts != []
    assert [
        frozenset((contact.body_a, contact.body_b)) for contact in rejected.contacts
    ] == [frozenset((contact.body_a, contact.body_b)) for contact in expected_contacts]
