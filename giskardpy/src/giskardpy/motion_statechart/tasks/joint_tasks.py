from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field, dataclass

import numpy as np
from typing_extensions import List

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.exceptions import EmptyGoalStateError
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.graph_node import ConvergingTask
from giskardpy.motion_statechart.monitors.joint_monitors import (
    CloseToJointConfigurations,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
)


@dataclass(eq=False, repr=False)
class JointPositionList(ConvergingTask):
    """
    Moves the robot to a given joint position.
    """

    goal_state: JointState = field(kw_only=True)
    """
    The goal joint state.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    If all joint position errors are smaller than this threshold, the task's observation
    state is true.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    The weight of this task.
    """

    max_velocity: float = field(default=1.0, kw_only=True)
    """
    The maximum velocity of the joints.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build one equality constraint per joint of the goal state.

        :param context: Provides access to world model and kinematic expressions.
        :return: The artifacts of this task, whose error is the largest absolute joint
            position error, so the task succeeds only once every joint is within the
            threshold.
        """
        if len(self.goal_state) == 0:
            raise EmptyGoalStateError(node=self)

        artifacts = NodeArtifacts()
        errors = []
        for connection, target in self.goal_state.items():
            current = connection.dof.variables.position
            target = self.apply_limits_to_target(target, connection)
            velocity = self.apply_limits_to_velocity(self.max_velocity, connection)
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = sm.shortest_angular_distance(current, target)
            else:
                error = target - current
            artifacts.constraints.add_equality_constraint(
                name=str(connection.name),
                reference_velocity=velocity,
                equality_bound=error,
                quadratic_weight=self.weight,
                task_expression=current,
            )
            errors.append(sm.abs(error))
        artifacts.error = sm.max(sm.Vector(errors))
        return artifacts

    def apply_limits_to_target(
        self, target: float, connection: ActiveConnection1DOF
    ) -> sm.Scalar:
        ul_pos = connection.dof.limits.upper.position
        ll_pos = connection.dof.limits.lower.position
        if ll_pos is not None:
            target = sm.limit(target, ll_pos, ul_pos)
        return target

    def apply_limits_to_velocity(
        self, velocity: float, connection: ActiveConnection1DOF
    ) -> sm.Scalar:
        ul_vel = connection.dof.limits.upper.velocity
        ll_vel = connection.dof.limits.lower.velocity
        return sm.limit(velocity, ll_vel, ul_vel)


@dataclass(eq=False, repr=False)
class JointVelocityLimit(Task):
    """
    Limits the velocity of a set of joints to a maximum value.

    The joint-space equivalent of
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`:
    enforcement is performed by adding a real inequality constraint to the optimizer
    for each given joint, not by tuning a reference/normalization velocity.
    """

    connections: List[ActiveConnection1DOF] = field(kw_only=True)
    """
    The joints whose velocity is constrained.
    """

    max_velocity: float = field(default=0.1, kw_only=True)
    """
    Maximum allowed velocity (in rad/s or m/s, per joint).
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Optimization weight determining how strongly the velocity limit is enforced.

    Defaults to the same weight as its Cartesian equivalents
    (:class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`,
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianRotationVelocityLimit`):
    a velocity *limit* is a physical constraint like a joint limit, not a goal, so it
    is weighted above collision avoidance rather than below it like goal tasks (e.g.
    :class:`JointPositionList`).
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        velocities = []
        for connection in self.connections:
            position = connection.dof.variables.position
            artifacts.constraints.add_velocity_constraint(
                upper_velocity_limit=self.max_velocity,
                lower_velocity_limit=-self.max_velocity,
                quadratic_weight=self.weight,
                task_expression=position,
                velocity_limit=self.max_velocity,
                name=str(connection.name),
            )
            velocities.append(connection.dof.variables.velocity)
        artifacts.observation = sm.logic_all(
            sm.abs(sm.Vector(velocities)) <= self.max_velocity
        )
        return artifacts


# %% paths planned at runtime


@dataclass(eq=False, repr=False)
class JointWaypointPath(JointPositionList, ABC):
    """
    Moves a group of connections through waypoints that are planned only once the task
    starts, from wherever the connections are at that moment.

    The error is the distance still to cover along the waypoints, so moving along the
    path counts as approaching the goal.
    """

    connections: list[ActiveConnection1DOF] = field(kw_only=True)
    """
    The connections moved along the waypoints.
    """

    waypoint_threshold: float = field(default=0.05, kw_only=True)
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

    def __post_init__(self):
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
    def plan_waypoints(self, context: MotionStatechartContext) -> list[np.ndarray]:
        """
        Plan the path from the positions the connections are at now.

        :param context: The context the task runs in.
        :return: At least one waypoint, each giving positions of :attr:`connections`;
            the last one is the goal.
        """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        context.float_variable_data.register_expression(self._goal)
        context.float_variable_data.register_expression(self._remaining_path_length)
        artifacts = super().build_artifacts(context)
        artifacts.error = artifacts.error + self._remaining_path_length
        return artifacts

    def on_start(self, context: MotionStatechartContext):
        self._waypoints = self.plan_waypoints(context)
        self._next_waypoint = 0
        self._aim_at_next_waypoint(context)

    def on_tick(self, context: MotionStatechartContext) -> None:
        if self._next_waypoint == len(self._waypoints) - 1:
            return
        positions = np.array([connection.position for connection in self.connections])
        waypoint = self._waypoints[self._next_waypoint]
        if np.max(np.abs(positions - waypoint)) <= self.waypoint_threshold:
            self._next_waypoint += 1
            self._aim_at_next_waypoint(context)

    def _aim_at_next_waypoint(self, context: MotionStatechartContext) -> None:
        """
        Set the goal to the waypoint currently aimed at and the remaining path length to
        the length of the path behind it.
        """
        remaining = self._waypoints[self._next_waypoint :]
        remaining_length = sum(
            float(np.max(np.abs(later - earlier)))
            for earlier, later in zip(remaining, remaining[1:])
        )
        context.float_variable_data.set_value(self._goal, remaining[0])
        context.float_variable_data.set_value(
            self._remaining_path_length, remaining_length
        )


@dataclass(eq=False, repr=False)
class FixedJointWaypointPath(JointWaypointPath):
    """
    Moves the connections through waypoints that are known when the task is built, such
    as a path planned outside of the controller.
    """

    waypoints: list[list[float]] = field(kw_only=True)
    """
    The positions of :attr:`connections` at every waypoint; the last one is the goal.
    """

    def plan_waypoints(self, context: MotionStatechartContext) -> list[np.ndarray]:
        return [np.array(waypoint) for waypoint in self.waypoints]


@dataclass(eq=False, repr=False)
class RetractToLastSafeJointConfiguration(JointWaypointPath):
    """
    Moves the connections a :class:`CloseToJointConfigurations` monitor watches back to
    the last positions it found safe.
    """

    monitor: CloseToJointConfigurations = field(kw_only=True)
    """
    The monitor that remembers the last safe positions.
    """

    connections: list[ActiveConnection1DOF] = field(init=False)
    """
    The connections the monitor watches.
    """

    def __post_init__(self):
        self.connections = self.monitor.configurations.connections
        super().__post_init__()

    def plan_waypoints(self, context: MotionStatechartContext) -> list[np.ndarray]:
        return [self.monitor.last_safe_positions]
