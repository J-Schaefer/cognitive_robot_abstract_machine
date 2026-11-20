"""
Tests of the joint paths planned once they start, and of retracing the joint states of
the last control cycles.
"""

from __future__ import annotations

import numpy as np
import pytest
from cramph.composites import Parallel, Sequence, StoppedWhenTrue, TryInOrder
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.joint_monitors import RecentJointStates
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
    RetraceRecentJointStates,
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
    """
    The recorder holds the positions of exactly its last control cycles, in order.
    """
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
    """
    Retracing stays on the recorded path around a corner and ends at its oldest state.
    """
    path = _corner_path(prismatic_bot2)
    recorder = RecentJointStates(connections=path.connections, control_cycles=1000)
    retrace = RetraceRecentJointStates(monitor=recorder)
    trip = CountTicks(ticks=25, name="trip")
    goal = _stopped_then(
        trip, Parallel(nodes=[path, recorder], minimum_success=1), retrace
    )
    executor = _compile(prismatic_bot2, goal)
    for _ in range(trip.ticks + 2):
        if retrace.life_cycle_state != LifeCycleValue.NOT_STARTED:
            break
        executor.tick()
    assert retrace.life_cycle_state == LifeCycleValue.RUNNING
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
    """
    A waypoint path comes close to every waypoint and reaches the last one.
    """
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
    assert path.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert all(
        distance <= path.waypoint_threshold for distance in closest_approach[:-1]
    )
    assert closest_approach[-1] <= path.threshold


def test_waypoint_path_plans_from_where_it_starts(prismatic_bot2: World):
    """
    A waypoint path plans from where the joints are once it starts.
    """
    connections = list(prismatic_bot2.connections)
    first_motion = JointPositionList(
        goal_state=JointState.from_lists(connections, [0.3, -0.2]), name="first"
    )
    path = FixedWaypointPath(connections=connections, waypoints=[np.zeros(2)])
    sequence = Sequence(nodes=[first_motion, path])
    _compile(prismatic_bot2, sequence).tick_until_end(timeout=500)

    assert path.planned_from == pytest.approx([0.3, -0.2], abs=first_motion.threshold)
