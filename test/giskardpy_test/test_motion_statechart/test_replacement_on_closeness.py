"""
Tests for leaving a motion that comes close to forbidden joint configurations: the
closeness monitor, and the waypoint paths that retract and plan anew once it stopped the
motion.
"""

from __future__ import annotations

import numpy as np
import pytest
from cramph.composites import Parallel, Sequence, StoppedWhenTrue, TryInOrder
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.joint_monitors import (
    CloseToJointConfigurations,
    JointConfigurationSet,
    RecentJointStates,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
    RetraceRecentJointStates,
    RetractToLastSafeJointConfiguration,
)
from semantic_digital_twin.world import World

from .fixed_waypoint_path import FixedWaypointPath

# %% helpers


def _compile(world: World, *nodes: StatechartNode) -> StatechartExecutor:
    """
    :return: An executor that compiled a chart of the given nodes, ending once the
        first of them ended.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    for node in nodes:
        statechart.add_node(node=node)
    end = EndMotion()
    end.start_condition = nodes[0].is_terminated
    statechart.add_node(node=end)
    executor.compile(statechart=statechart)
    return executor


def _stopped_then(
    monitor: StatechartNode, monitored_node: StatechartNode, then: StatechartNode
) -> TryInOrder:
    """
    :return: The monitored node, stopped once the monitor observes True, and the node
        that runs once it was stopped.
    """
    return TryInOrder(
        nodes=[StoppedWhenTrue(monitor=monitor, monitored_node=monitored_node), then]
    )


def _goal_ended_on_its_observation(goal: StatechartNode) -> None:
    """
    Let the goal succeed once it observes its goal, as the coraplex executable ends a
    chart.
    """
    goal.success_condition = goal.observes_true


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
        cruise_velocity=0.2,
    )
    return monitor, motion


def test_closeness_monitor_stops_the_motion_before_the_forbidden_position(
    prismatic_bot: World,
):
    monitor, motion = _approach(prismatic_bot, forbidden=0.5, goal=0.9)
    goal = StoppedWhenTrue(monitor=monitor, monitored_node=motion)
    _goal_ended_on_its_observation(goal)
    _compile(prismatic_bot, goal).tick_until_end(timeout=500)

    (connection,) = prismatic_bot.connections
    assert motion.life_cycle_state == LifeCycleValues.INTERRUPTED
    assert connection.position < 0.5


def test_closeness_monitor_remembers_the_last_safe_position(prismatic_bot: World):
    monitor, motion = _approach(prismatic_bot, forbidden=0.5, goal=0.9)
    goal = StoppedWhenTrue(monitor=monitor, monitored_node=motion)
    _goal_ended_on_its_observation(goal)
    _compile(prismatic_bot, goal).tick_until_end(timeout=500)

    distance = monitor.configurations.distance_to(monitor.last_safe_positions)
    assert monitor.safe_distance <= distance < monitor.safe_distance + 0.1


def test_retract_returns_to_the_last_safe_position(prismatic_bot: World):
    monitor, motion = _approach(prismatic_bot, forbidden=0.5, goal=0.9)
    retract = RetractToLastSafeJointConfiguration(monitor=monitor)
    goal = _stopped_then(monitor, motion, retract)
    _compile(prismatic_bot, goal).tick_until_end(timeout=500)

    (connection,) = prismatic_bot.connections
    assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert connection.position == pytest.approx(
        monitor.last_safe_positions[0], abs=retract.threshold
    )


# %% retracing the recent joint states


def _corner_path(world: World) -> FixedWaypointPath:
    """
    :return: A path of both connections of the world around a corner.
    """
    return FixedWaypointPath(
        connections=list(world.connections),
        waypoints=[np.array([0.4, 0.0]), np.array([0.4, 0.4])],
    )


def _positions_per_control_cycle(
    executor: StatechartExecutor, path
) -> list[np.ndarray]:
    """
    :return: The positions of the path's connections before the first and after every
        control cycle until the motion ends.
    """
    visited = [np.array([connection.position for connection in path.connections])]
    while not executor.statechart.is_ended():
        executor.tick()
        visited.append(
            np.array([connection.position for connection in path.connections])
        )
    return visited


def test_recent_joint_states_keep_the_positions_of_the_last_control_cycles(
    prismatic_bot2: World,
):
    path = _corner_path(prismatic_bot2)
    recorder = RecentJointStates(connections=path.connections, control_cycles=10)
    motion = Parallel(nodes=[path, recorder], minimum_success=1)
    motion.success_condition = motion.observes_true
    visited = _positions_per_control_cycle(_compile(prismatic_bot2, motion), path)

    assert len(recorder.states) == recorder.control_cycles
    assert any(
        np.array_equal(np.array(recorder.states), np.array(visited[end - 10 : end]))
        for end in range(10, len(visited) + 1)
    )


def test_retrace_follows_the_recent_joint_states_back(prismatic_bot2: World):
    path = _corner_path(prismatic_bot2)
    recorder = RecentJointStates(connections=path.connections, control_cycles=1000)
    retrace = RetraceRecentJointStates(monitor=recorder)
    trip = CountTicks(ticks=25, name="trip")
    goal = _stopped_then(
        trip, Parallel(nodes=[path, recorder], minimum_success=1), retrace
    )
    executor = _compile(prismatic_bot2, goal)
    for _ in range(trip.ticks + 2):
        if retrace.life_cycle_state != LifeCycleValues.NOT_STARTED:
            break
        executor.tick()
    assert retrace.life_cycle_state == LifeCycleValues.RUNNING
    recorded = np.array(retrace.plan_waypoints())
    retraced = np.array(_positions_per_control_cycle(executor, path))

    distance_to_recorded_path = [
        np.min(np.max(np.abs(recorded - positions), axis=1)) for positions in retraced
    ]
    assert recorded[0][1] > 0.0
    assert max(distance_to_recorded_path) <= retrace.waypoint_threshold
    assert retraced[-1] == pytest.approx(recorder.states[0], abs=retrace.threshold)


# %% waypoint paths


def test_waypoint_path_passes_every_waypoint(prismatic_bot2: World):
    waypoints = [np.array([0.4, 0.0]), np.array([0.4, 0.4]), np.array([0.0, 0.4])]
    path = FixedWaypointPath(
        connections=list(prismatic_bot2.connections), waypoints=waypoints
    )
    path.success_condition = path.observes_true
    executor = _compile(prismatic_bot2, path)
    visited = []
    while not executor.statechart.is_ended():
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
    _compile(prismatic_bot2, sequence).tick_until_end(timeout=500)

    assert path.planned_from == pytest.approx([0.3, -0.2], abs=first_motion.threshold)
