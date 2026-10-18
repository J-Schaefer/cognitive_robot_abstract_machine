"""
Tests for leaving a motion that comes close to forbidden joint configurations: the
closeness monitor, the goal replacing the motion, and the waypoint paths that retract
and plan anew.
"""

from __future__ import annotations

import numpy as np
import pytest
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.joint_monitors import (
    CloseToJointConfigurations,
    JointConfigurationSet,
)
from giskardpy.motion_statechart.monitors.payload_monitors import CountControlCycles
from giskardpy.motion_statechart.monitors.templates import ReplacedWhenTrue
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstTrueNode,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
    RetractToLastSafeJointConfiguration,
)
from krrood.symbolic_math.symbolic_math import trinary_logic_not
from semantic_digital_twin.world import World

from .fixed_waypoint_path import FixedWaypointPath

# %% helpers


def _compile(world: World, *nodes) -> Executor:
    """
    :return: An executor that compiled a chart of the given nodes, ending once the
        first of them ended.
    """
    msc = MotionStatechart()
    for node in nodes:
        msc.add_node(node)
    end = EndMotion()
    end.start_condition = nodes[0].is_terminated
    msc.add_node(end)
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=msc)
    return executor


def _goal_ended_on_its_observation(goal) -> None:
    """
    Let the goal end once it observes its goal, as the coraplex executable ends a chart.
    """
    goal.end_condition = goal.observation_variable


# %% joint configuration set


def test_distance_counts_only_the_relevant_connections(prismatic_bot2: World):
    configurations = JointConfigurationSet(
        connections=list(prismatic_bot2.connections),
        positions=np.array([[0.5, 0.5], [-0.5, 0.9]]),
        relevant_connections=np.array([[True, True], [True, False]]),
    )

    assert configurations.distance_to(np.array([-0.4, -0.9])) == pytest.approx(0.1)


def test_distance_to_an_empty_set_is_infinite(prismatic_bot2: World):
    configurations = JointConfigurationSet(
        connections=list(prismatic_bot2.connections),
        positions=np.empty((0, 2)),
        relevant_connections=np.empty((0, 2), dtype=bool),
    )

    assert configurations.distance_to(np.zeros(2)) == np.inf


# %% closeness monitor and retracting


def _approach(world: World, forbidden: float, goal: float):
    """
    :return: The monitor watching for the forbidden position, and the motion of the
        only connection of the world towards the goal through it.
    """
    (connection,) = world.connections
    monitor = CloseToJointConfigurations(
        configurations=JointConfigurationSet(
            connections=[connection],
            positions=np.array([[forbidden]]),
            relevant_connections=np.array([[True]]),
        ),
        threshold=0.1,
        safe_distance=0.3,
    )
    motion = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal}),
        name="approach",
        max_velocity=0.2,
    )
    return monitor, motion


def test_closeness_monitor_stops_the_motion_before_the_forbidden_position(
    prismatic_bot: World,
):
    monitor, motion = _approach(prismatic_bot, forbidden=0.5, goal=0.9)
    goal = ReplacedWhenTrue(
        monitor=monitor, monitored_node=motion, replacement=ConstTrueNode()
    )
    _goal_ended_on_its_observation(goal)
    _compile(prismatic_bot, goal).tick_until_end(timeout=500)

    (connection,) = prismatic_bot.connections
    assert motion.life_cycle_state == LifeCycleValues.FAILED
    assert connection.position < 0.5


def test_closeness_monitor_remembers_the_last_safe_position(prismatic_bot: World):
    monitor, motion = _approach(prismatic_bot, forbidden=0.5, goal=0.9)
    goal = ReplacedWhenTrue(
        monitor=monitor, monitored_node=motion, replacement=ConstTrueNode()
    )
    _goal_ended_on_its_observation(goal)
    _compile(prismatic_bot, goal).tick_until_end(timeout=500)

    distance = monitor.configurations.distance_to(monitor.last_safe_positions)
    assert monitor.safe_distance <= distance < monitor.safe_distance + 0.1


def test_retract_returns_to_the_last_safe_position(prismatic_bot: World):
    monitor, motion = _approach(prismatic_bot, forbidden=0.5, goal=0.9)
    retract = RetractToLastSafeJointConfiguration(monitor=monitor)
    goal = ReplacedWhenTrue(monitor=monitor, monitored_node=motion, replacement=retract)
    _goal_ended_on_its_observation(goal)
    _compile(prismatic_bot, goal).tick_until_end(timeout=500)

    (connection,) = prismatic_bot.connections
    assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert connection.position == pytest.approx(
        monitor.last_safe_positions[0], abs=retract.threshold
    )


# %% replacement goal


def test_replacement_stays_dormant_while_the_monitor_does_not_fire(
    prismatic_bot: World,
):
    replacement = ConstTrueNode()
    goal = ReplacedWhenTrue(
        monitor=CountControlCycles(control_cycles=99, name="trip"),
        monitored_node=ConstTrueNode(name="work"),
        replacement=replacement,
    )
    _goal_ended_on_its_observation(goal)
    _compile(prismatic_bot, goal).tick_until_end(timeout=50)

    assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert replacement.life_cycle_state == LifeCycleValues.NOT_STARTED


def test_replacement_reaching_its_goal_succeeds_the_goal(prismatic_bot: World):
    replacement = ConstTrueNode()
    goal = ReplacedWhenTrue(
        monitor=CountControlCycles(control_cycles=2, name="trip"),
        monitored_node=CountControlCycles(control_cycles=99, name="work"),
        replacement=replacement,
    )
    _goal_ended_on_its_observation(goal)
    _compile(prismatic_bot, goal).tick_until_end(timeout=50)

    assert goal.monitored_node.life_cycle_state == LifeCycleValues.FAILED
    assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_replacement_ending_short_of_its_goal_fails_the_goal(prismatic_bot: World):
    replacement = CountControlCycles(control_cycles=99, name="replacement")
    replacement.end_condition = trinary_logic_not(replacement.observation_variable)
    goal = ReplacedWhenTrue(
        monitor=CountControlCycles(control_cycles=2, name="trip"),
        monitored_node=CountControlCycles(control_cycles=99, name="work"),
        replacement=replacement,
    )
    goal.end_condition = trinary_logic_not(goal.observation_variable)
    _compile(prismatic_bot, goal).tick_until_end(timeout=50)

    assert replacement.life_cycle_state == LifeCycleValues.FAILED
    assert goal.life_cycle_state == LifeCycleValues.FAILED


# %% waypoint paths


def test_waypoint_path_passes_every_waypoint(prismatic_bot2: World):
    waypoints = [np.array([0.4, 0.0]), np.array([0.4, 0.4]), np.array([0.0, 0.4])]
    path = FixedWaypointPath(
        connections=list(prismatic_bot2.connections), waypoints=waypoints
    )
    path.end_condition = path.observation_variable
    executor = _compile(prismatic_bot2, path)
    visited = []
    while not executor.motion_statechart.is_end_motion():
        executor.tick()
        visited.append([connection.position for connection in path.connections])

    closest_approach = [
        np.min(np.max(np.abs(np.array(visited) - waypoint), axis=1))
        for waypoint in waypoints
    ]
    assert path.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert all(
        distance <= path.waypoint_threshold for distance in closest_approach[:-1]
    )
    assert closest_approach[-1] <= path.threshold


def test_waypoint_path_plans_from_where_it_starts(prismatic_bot2: World):
    connections = list(prismatic_bot2.connections)
    first_motion = JointPositionList(
        goal_state=JointState.from_lists(connections, [0.3, -0.2]), name="first"
    )
    path = FixedWaypointPath(connections=connections, waypoints=[np.zeros(2)])
    sequence = Sequence(nodes=[first_motion, path])
    sequence.end_condition = sequence.observation_variable
    _compile(prismatic_bot2, sequence).tick_until_end(timeout=500)

    assert path.planned_from == pytest.approx([0.3, -0.2], abs=first_motion.threshold)
