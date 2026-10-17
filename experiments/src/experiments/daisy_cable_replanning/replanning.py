from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from itertools import pairwise
from math import ceil
from pathlib import Path

import numpy as np
from coraplex.plans.plan_node import ActionNode, MotionNode
from coraplex.plans.plan_transformation import PlanTransformation
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from experiments.daisy_cable_replanning.collisions import (
    ActiveMotionTracker,
    ArmCollisionPairs,
    IllegalJointStateDatabase,
    MotionLabel,
)
from experiments.daisy_cable_replanning.exceptions import NoJointPathFound
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.joint_monitors import (
    CloseToJointConfigurations,
    JointConfigurationSet,
)
from giskardpy.motion_statechart.monitors.templates import ReplacedWhenTrue
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointWaypointPath,
    RetractToLastSafeJointConfiguration,
)
from krrood.adapters.json_serializer import to_json
from ompl import base as ompl_base
from ompl import geometric as ompl_geometric
from semantic_digital_twin.collision_checking.collision_matrix import CollisionMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Self

# %% parameters


@dataclass(frozen=True)
class ReplanningParameters:
    """
    When a motion gives way to a planned path, and how that path is planned.
    """

    trigger_distance: float = 0.15
    """
    Joint distance to an illegal joint state, in radians, below which the motion is
    left.
    """

    safe_distance: float = 0.3
    """
    Joint distance to every illegal joint state, in radians, at which a joint state
    counts as safe to retract to.
    """

    clearance: float = 0.1
    """
    Joint distance, in radians, the planned path keeps from every illegal joint state.
    """

    collision_margin: float = 0.005
    """
    Distance in metres below which a body of the planned arm counts as colliding.
    """

    planning_time: float = 10.0
    """
    How long the planner may search for a path, in seconds.
    """

    waypoint_spacing: float = 0.1
    """
    Largest joint difference between consecutive waypoints of the planned path, in
    radians.
    """


# %% collision probing


@dataclass
class ProbedJointState:
    """
    Joint positions written into a world without telling anyone observing its state, so
    the collision detector can be asked about them, and restored on exit.
    """

    world: World
    """
    The world the positions are written into.
    """

    connections: list[ActiveConnection1DOF]
    """
    The connections whose positions are probed.
    """

    _columns: list[int] = field(init=False)
    """
    The columns of the connections in the state data.
    """

    _original_positions: np.ndarray = field(init=False)
    """
    The positions the connections had before probing.
    """

    def __post_init__(self):
        self._columns = self.world.state.column_indices(
            [connection.dof for connection in self.connections]
        )

    def __enter__(self) -> Self:
        self._original_positions = self.world.state.positions[self._columns].copy()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self._write(self._original_positions)

    def is_collision_free(
        self, positions: np.ndarray, collision_matrix: CollisionMatrix
    ) -> bool:
        """
        :param positions: Positions of :attr:`connections` to probe.
        :param collision_matrix: The pairs that may not collide, each with the distance
            below which it counts as colliding.
        :return: Whether no pair collides at the positions.
        """
        self._write(positions)
        collision_detector = self.world.collision_manager.collision_detector
        return not collision_detector.check_collisions(collision_matrix).any()

    def _write(self, positions: np.ndarray) -> None:
        """
        Write the positions and bring the collision detector up to date with them.
        """
        self.world.state.positions[self._columns] = positions
        self.world.collision_manager.collision_detector.world_state_updater.on_state_change()


# %% planning


@dataclass
class JointSpacePlanner:
    """
    Plans paths of a group of joints with OMPL's RRT-Connect that are free of
    collisions and keep clear of a set of illegal joint configurations, with every other
    joint held where it is.
    """

    world: World
    """
    The world the joints are in.
    """

    collision_matrix: CollisionMatrix
    """
    The pairs of bodies that may not collide along the path.
    """

    illegal_configurations: JointConfigurationSet
    """
    The joint configurations the path keeps clear of.
    """

    parameters: ReplanningParameters
    """
    How the path is planned.
    """

    def plan(
        self, connections: list[ActiveConnection1DOF], goal: np.ndarray
    ) -> list[np.ndarray]:
        """
        :param connections: The joints to move.
        :param goal: Their positions at the goal.
        :return: Waypoints from their current positions to the goal, no further apart
            than :attr:`ReplanningParameters.waypoint_spacing`.
        :raises NoJointPathFound: If no path is found within the planning time.
        """
        start = np.array([connection.position for connection in connections])
        held_configuration = self.illegal_configurations.current_positions()
        planned_columns = [
            self.illegal_configurations.connections.index(connection)
            for connection in connections
        ]
        space = self._state_space(connections)
        setup = ompl_geometric.SimpleSetup(space)
        with ProbedJointState(world=self.world, connections=connections) as probe:

            def is_valid(state) -> bool:
                positions = np.array(
                    [state[index] for index in range(len(connections))]
                )
                configuration = held_configuration.copy()
                configuration[planned_columns] = positions
                return self.illegal_configurations.distance_to(
                    configuration
                ) >= self.parameters.clearance and probe.is_collision_free(
                    positions, self.collision_matrix
                )

            setup.setStateValidityChecker(is_valid)
            setup.setStartAndGoalStates(
                self._ompl_state(space, start), self._ompl_state(space, goal)
            )
            setup.setPlanner(ompl_geometric.RRTConnect(setup.getSpaceInformation()))
            setup.getSpaceInformation().setStateValidityCheckingResolution(0.005)
            solved = setup.solve(self.parameters.planning_time)
            if not (solved and setup.haveExactSolutionPath()):
                raise NoJointPathFound(
                    start=start.tolist(),
                    goal=goal.tolist(),
                    planning_time=self.parameters.planning_time,
                )
            setup.simplifySolution(self.parameters.planning_time)
        return self._waypoints_along(setup.getSolutionPath(), len(connections))

    def _waypoints_along(self, path, dimension: int) -> list[np.ndarray]:
        """
        :return: The states of the OMPL path, interpolated to the waypoint spacing.
        """
        states = [
            np.array([state[index] for index in range(dimension)])
            for state in path.getStates()
        ]
        largest_steps = [
            np.max(np.abs(later - earlier)) for earlier, later in pairwise(states)
        ]
        path.interpolate(
            max(
                len(states),
                ceil(sum(largest_steps) / self.parameters.waypoint_spacing) + 1,
            )
        )
        return [
            np.array([state[index] for index in range(dimension)])
            for state in path.getStates()
        ]

    @staticmethod
    def _state_space(
        connections: list[ActiveConnection1DOF],
    ) -> ompl_base.RealVectorStateSpace:
        """
        :return: The space of the joints' positions, bounded by their position limits.
        """
        space = ompl_base.RealVectorStateSpace(len(connections))
        bounds = ompl_base.RealVectorBounds(len(connections))
        for index, connection in enumerate(connections):
            bounds.setLow(index, connection.dof.limits.lower.position)
            bounds.setHigh(index, connection.dof.limits.upper.position)
        space.setBounds(bounds)
        return space

    @staticmethod
    def _ompl_state(space: ompl_base.RealVectorStateSpace, positions: np.ndarray):
        """
        :return: An OMPL state of the space holding the positions.
        """
        state = space.allocState()
        for index, position in enumerate(positions):
            state[index] = float(position)
        return state


# %% replanning log


@dataclass
class ReplanningEvent:
    """
    A motion that was left because it came close to an illegal joint state, and the
    path that replaced it.
    """

    trial: int
    """
    Number of the trial the motion was left in.
    """

    motion: MotionLabel | None
    """
    The motion that was left.
    """

    retracted_to: list[float]
    """
    Positions of the planned joints after retracting, where the path starts.
    """

    waypoints: list[list[float]]
    """
    The waypoints of the planned path.
    """


@dataclass
class ReplanningLog:
    """
    Appends every replanning of a trial to a file, one JSON encoded
    :class:`ReplanningEvent` per line.
    """

    path: Path
    """
    The file the events are appended to.
    """

    trial: int
    """
    Number of the trial.
    """

    motion_tracker: ActiveMotionTracker
    """
    Names the motion that was left.
    """

    def record(self, waypoints: list[np.ndarray]) -> None:
        """
        :param waypoints: The waypoints of the planned path, the first one being where
            the path starts.
        """
        event = ReplanningEvent(
            trial=self.trial,
            motion=self.motion_tracker.active_motion,
            retracted_to=waypoints[0].tolist(),
            waypoints=[waypoint.tolist() for waypoint in waypoints],
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as log:
            log.write(json.dumps(to_json(event)) + "\n")


# %% planned path


@dataclass(eq=False, repr=False)
class PlannedJointPathToPose(JointWaypointPath):
    """
    Moves an arm along a path planned, once it starts, from where the arm is to a joint
    configuration that puts its tool frame at a goal pose.
    """

    planner: JointSpacePlanner = field(kw_only=True)
    """
    Plans the path.
    """

    root: Body = field(kw_only=True)
    """
    The body the arm is mounted below.
    """

    tool_frame: Body = field(kw_only=True)
    """
    The body that is brought to the goal pose.
    """

    goal_pose: Pose = field(kw_only=True)
    """
    Where the tool frame is brought.
    """

    replanning_log: ReplanningLog = field(kw_only=True)
    """
    Where the planned path is recorded.
    """

    def plan_waypoints(self, context: MotionStatechartContext) -> list[np.ndarray]:
        goal_positions = context.world.compute_inverse_kinematics(
            root=self.root,
            tip=self.tool_frame,
            target=self.goal_pose.homogeneous_matrix,
        )
        goal = np.array(
            [goal_positions[connection.dof] for connection in self.connections]
        )
        waypoints = self.planner.plan(self.connections, goal)
        self.replanning_log.record(waypoints)
        return waypoints


# %% replanned motion


@dataclass
class ReplannedToolCenterPointMotion(MoveToolCenterPointMotion):
    """
    Moves the tool center point like :class:`MoveToolCenterPointMotion`, but leaves the
    motion as soon as the arms come close to an illegal joint state, retracts to the
    last safe joint state and reaches the target along a path planned around the
    illegal joint states instead.
    """

    illegal_joint_states: IllegalJointStateDatabase = field(kw_only=True)
    """
    The joint states to stay away from.
    """

    parameters: ReplanningParameters = field(kw_only=True)
    """
    When the motion is left and how the replacing path is planned.
    """

    replanning_log: ReplanningLog = field(kw_only=True)
    """
    Where the replacing paths are recorded.
    """

    @classmethod
    def replacing(
        cls,
        motion: MoveToolCenterPointMotion,
        illegal_joint_states: IllegalJointStateDatabase,
        parameters: ReplanningParameters,
        replanning_log: ReplanningLog,
    ) -> ReplannedToolCenterPointMotion:
        """
        :return: A replanned motion with the parameters of the given motion.
        """
        return cls(
            **{
                motion_field.name: getattr(motion, motion_field.name)
                for motion_field in fields(motion)
                if motion_field.init
            },
            illegal_joint_states=illegal_joint_states,
            parameters=parameters,
            replanning_log=replanning_log,
        )

    @property
    def _motion_chart(self) -> ReplacedWhenTrue:
        illegal_configurations = self.illegal_joint_states.joint_configurations(
            self.world
        )
        monitor = CloseToJointConfigurations(
            name="close to illegal joint state",
            configurations=illegal_configurations,
            threshold=self.parameters.trigger_distance,
            safe_distance=self.parameters.safe_distance,
        )
        planner = JointSpacePlanner(
            world=self.world,
            collision_matrix=ArmCollisionPairs(
                world=self.world, robot=self.robot
            ).collision_matrix_of(self.arm, self.parameters.collision_margin),
            illegal_configurations=illegal_configurations,
            parameters=self.parameters,
        )
        return ReplacedWhenTrue(
            name="MoveTCP or replanned path",
            monitor=monitor,
            monitored_node=super()._motion_chart,
            replacement=Sequence(
                name="retract and follow planned path",
                nodes=[
                    RetractToLastSafeJointConfiguration(
                        name="retract to last safe joint state", monitor=monitor
                    ),
                    PlannedJointPathToPose(
                        name="planned joint path",
                        connections=self.arm.active_connections,
                        planner=planner,
                        root=self.robot.root,
                        tool_frame=self.arm.end_effector.tool_frame,
                        goal_pose=self.target,
                        replanning_log=self.replanning_log,
                    ),
                ],
            ),
        )


# %% plan transformation


@dataclass
class ReplanningTransformation(PlanTransformation[ActionDescription]):
    """
    Replaces the tool center point motions of the actions illegal joint states were
    recorded in by :class:`ReplannedToolCenterPointMotion`.
    """

    illegal_joint_states: IllegalJointStateDatabase = field(kw_only=True)
    """
    The joint states to stay away from.
    """

    parameters: ReplanningParameters = field(kw_only=True)
    """
    When a motion is left and how the replacing path is planned.
    """

    replanning_log: ReplanningLog = field(kw_only=True)
    """
    Where the replacing paths are recorded.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return type(plan_node.action).__name__ in {
            state.motion.action
            for state in self.illegal_joint_states.illegal_joint_states
        }

    def apply(self, plan_node: ActionNode) -> None:
        for node in plan_node.descendants:
            if not isinstance(node, MotionNode) or type(node.designator) is not (
                MoveToolCenterPointMotion
            ):
                continue
            node.designator = ReplannedToolCenterPointMotion.replacing(
                node.designator,
                self.illegal_joint_states,
                self.parameters,
                self.replanning_log,
            )
            node.designator.plan_node = node
