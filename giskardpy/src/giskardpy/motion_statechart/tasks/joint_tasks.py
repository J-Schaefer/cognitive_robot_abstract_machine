from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
from giskardpy.motion_statechart.weights import DefaultWeights, VelocityScale
from giskardpy.motion_statechart.exceptions import EmptyGoalStateError
from giskardpy.motion_statechart.graph_node import MotionNodeArtifacts, Task
from giskardpy.motion_statechart.graph_node import ConvergingTask
from giskardpy.motion_statechart.monitors.joint_monitors import (
    CloseToJointConfigurations,
    RecentJointStates,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)


@dataclass(eq=False, repr=False, kw_only=True)
class JointPositionList(ConvergingTask):
    """
    Moves the robot to a given joint position.
    """

    goal_state: JointState
    """
    The goal joint state.
    """

    threshold: float = 0.01
    """
    If all joint position errors are smaller than this threshold, the task's observation
    state is true.
    """

    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """
    The weight of this task.
    """

    cruise_velocity: float = 2.0
    """
    Sets how fast the joints approach the goal, in rad/s or m/s; not strictly enforced.

    A joint never moves faster than its velocity limit.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build one equality constraint per joint of the goal state.

        :return: The artifacts of this task, whose error is the largest absolute joint
            position error, so the task succeeds only once every joint is within the
            threshold.
        """
        if len(self.goal_state) == 0:
            raise EmptyGoalStateError(node=self)

        artifacts = MotionNodeArtifacts()
        errors = []
        for connection, target in self.goal_state.items():
            current = connection.dof.variables.position
            target = self.apply_limits_to_target(target=target, connection=connection)
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = sm.shortest_angular_distance(current, target)
            else:
                error = target - current
            artifacts.constraints.add_equality_constraint(
                name=str(connection.name),
                cruise_velocity=self.cruise_velocity,
                velocity_scale=min(
                    connection.dof.limits.upper.velocity, VelocityScale.JOINT
                ),
                equality_bound=error,
                quadratic_weight=self.weight,
                task_expression=current,
            )
            errors.append(sm.abs(error))
        artifacts.error = sm.max(sm.Vector(errors))
        return artifacts

    def apply_limits_to_target(
        self, *, target: float, connection: ActiveConnection1DOF
    ) -> sm.Scalar:
        ul_pos = connection.dof.limits.upper.position
        ll_pos = connection.dof.limits.lower.position
        if ll_pos is not None:
            target = sm.limit(target, ll_pos, ul_pos)
        return target


@dataclass(eq=False, repr=False, kw_only=True)
class JointVelocityLimit(Task):
    """
    Limits the velocity of a set of joints to a maximum value.

    The joint-space equivalent of
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`:
    enforcement is performed by adding a real inequality constraint to the optimizer
    for each given joint, not by tuning a cruise velocity or a velocity scale.
    """

    connections: list[ActiveConnection1DOF]
    """
    The joints whose velocity is constrained.
    """

    max_velocity: float = 0.1
    """
    Maximum allowed velocity (in rad/s or m/s, per joint).
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Optimization weight determining how strongly the velocity limit is enforced.

    Defaults to the same weight as its Cartesian equivalents
    (:class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`,
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianRotationVelocityLimit`):
    a velocity *limit* is a physical constraint like a joint limit, not a goal, so it
    is weighted above collision avoidance rather than below it like goal tasks (e.g.
    :class:`JointPositionList`).
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()
        velocities = []
        for connection in self.connections:
            position = connection.dof.variables.position
            artifacts.constraints.add_velocity_constraint(
                upper_velocity_limit=self.max_velocity,
                lower_velocity_limit=-self.max_velocity,
                quadratic_weight=self.weight,
                task_expression=position,
                velocity_scale=connection.dof.limits.upper.velocity,
                name=str(connection.name),
            )
            velocities.append(connection.dof.variables.velocity)
        artifacts.observation = sm.logic_all(
            sm.abs(sm.Vector(velocities)) <= self.max_velocity
        )
        return artifacts


# %% paths planned at runtime


@dataclass(eq=False, repr=False, kw_only=True)
class JointWaypointPath(JointPositionList, ABC):
    """
    Moves a group of connections through waypoints that are planned only once the task
    starts, from wherever the connections are at that moment.

    The error is the distance still to cover along the waypoints, so moving along the
    path counts as approaching the goal.
    """

    connections: list[ActiveConnection1DOF]
    """
    The connections moved along the waypoints.
    """

    waypoint_threshold: float = 0.05
    """
    Largest position error at which a waypoint before the last one counts as passed.
    """

    goal_state: JointState = field(init=False)
    """
    The waypoint currently aimed at, whose targets are set while the task runs.
    """

    _goal: sm.Vector = field(init=False, repr=False)
    """
    The positions of the waypoint currently aimed at.
    """

    _remaining_path_length: sm.FloatVariable = field(init=False, repr=False)
    """
    The length of the path behind the waypoint currently aimed at, measured as the sum
    of the largest position differences between consecutive waypoints.
    """

    _waypoints: list[np.ndarray] = field(init=False, default_factory=list, repr=False)
    """
    The waypoints planned when the task started.
    """

    _next_waypoint: int = field(init=False, default=0, repr=False)
    """
    Index of the waypoint currently aimed at.
    """

    def __post_init__(self) -> None:
        """
        Create the variables of the waypoint aimed at and of the remaining path length.
        """
        super().__post_init__()
        self._goal = sm.Vector(
            [
                sm.FloatVariable(name=f"{self._node_id}/goal/{connection.name}")
                for connection in self.connections
            ]
        )
        self._remaining_path_length = sm.FloatVariable(
            name=f"{self._node_id}/remaining_path_length"
        )
        self.goal_state = JointState.from_lists(
            self.connections,
            [self._goal[index] for index in range(len(self.connections))],
        )

    @abstractmethod
    def plan_waypoints(self) -> list[np.ndarray]:
        """
        Plan the path from the positions the connections are at now.

        :return: At least one waypoint, each giving positions of :attr:`connections`;
            the last one is the goal.
        """

    def set_up(self) -> None:
        """
        Differs from :meth:`~cramph.node.StatechartNode.set_up` in registering the
        variables of the waypoint aimed at and of the remaining path length.
        """
        self.context.float_variable_data.register_expression(expression=self._goal)
        self.context.float_variable_data.register_expression(
            expression=self._remaining_path_length
        )

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Differs from :meth:`JointPositionList.build_artifacts` in adding the remaining
        path length to the error.
        """
        artifacts = super().build_artifacts()
        artifacts.error = artifacts.error + self._remaining_path_length
        return artifacts

    def on_start(self) -> None:
        """
        Differs from :meth:`~cramph.node.StatechartNode.on_start` in planning the
        waypoints and aiming at the first one.
        """
        self._waypoints = self.plan_waypoints()
        self._next_waypoint = 0
        self._aim_at_next_waypoint()

    def on_tick(self) -> None:
        """
        Differs from :meth:`~cramph.node.StatechartNode.on_tick` in aiming at the next
        waypoint once the current one before the last is passed.
        """
        if self._next_waypoint == len(self._waypoints) - 1:
            return
        positions = np.array([connection.position for connection in self.connections])
        waypoint = self._waypoints[self._next_waypoint]
        if np.max(np.abs(positions - waypoint)) <= self.waypoint_threshold:
            self._next_waypoint += 1
            self._aim_at_next_waypoint()

    def _aim_at_next_waypoint(self) -> None:
        """
        Set the goal to the waypoint currently aimed at and the remaining path length to
        the length of the path behind it.
        """
        remaining = self._waypoints[self._next_waypoint :]
        remaining_length = sum(
            float(np.max(np.abs(later - earlier)))
            for earlier, later in zip(remaining, remaining[1:])
        )
        self.context.float_variable_data.set_value(
            expression=self._goal, value=remaining[0]
        )
        self.context.float_variable_data.set_value(
            expression=self._remaining_path_length, value=remaining_length
        )


@dataclass(eq=False, repr=False, kw_only=True)
class FixedJointWaypointPath(JointWaypointPath):
    """
    Moves the connections through waypoints that are known when the task is built, such
    as a path planned outside of the controller.
    """

    waypoints: list[list[float]]
    """
    The positions of :attr:`connections` at every waypoint; the last one is the goal.
    """

    def plan_waypoints(self) -> list[np.ndarray]:
        return [np.array(waypoint) for waypoint in self.waypoints]


@dataclass(eq=False, repr=False, kw_only=True)
class RetractToLastSafeJointConfiguration(JointWaypointPath):
    """
    Moves the connections a :class:`CloseToJointConfigurations` monitor watches back to
    the last positions it found safe.
    """

    monitor: CloseToJointConfigurations
    """
    The monitor that remembers the last safe positions.
    """

    connections: list[ActiveConnection1DOF] = field(init=False)
    """
    The connections the monitor watches.
    """

    def __post_init__(self) -> None:
        """
        Take the connections from the monitor before creating the goal variables.
        """
        self.connections = self.monitor.configurations.connections
        super().__post_init__()

    def plan_waypoints(self) -> list[np.ndarray]:
        return [self.monitor.last_safe_positions]


@dataclass(eq=False, repr=False, kw_only=True)
class RetraceRecentJointStates(JointWaypointPath):
    """
    Moves the connections a :class:`RecentJointStates` monitor watches back through the
    positions it remembered, the latest first, to the earliest one.

    The waypoints are one control cycle apart, so the connections follow the path they
    came within :attr:`waypoint_threshold`.
    """

    monitor: RecentJointStates
    """
    The monitor that remembers the positions.
    """

    connections: list[ActiveConnection1DOF] = field(init=False)
    """
    The connections the monitor watches.
    """

    def __post_init__(self) -> None:
        """
        Take the connections from the monitor before creating the goal variables.
        """
        self.connections = self.monitor.connections
        super().__post_init__()

    def plan_waypoints(self) -> list[np.ndarray]:
        return list(reversed(self.monitor.states))
