from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum
from itertools import pairwise
from math import ceil, pi
from pathlib import Path

import numpy as np
from coraplex.datastructures.enums import InsertionPosition
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.plan_transformation import (
    PlanRewriting,
    PlanTransformation,
    RecoveryByInsertion,
    SequentialInsertion,
)
from cramph.composites import Parallel, Sequence, StoppedWhenTrue
from cramph.data_types import ObservationStateValue
from cramph.failure_handling import HandledOnFailure
from cramph.node import EndedByOwner, StatechartNode
from experiments.daisy_cable_replanning.collisions import (
    ActiveMotionTracker,
    ArmCollisionPairs,
    IllegalJointStateDatabase,
    MotionLabel,
)
from experiments.daisy_cable_replanning.exceptions import (
    NoJointPathFound,
    NoValidGoalConfiguration,
    RejectedGoalConfiguration,
)
from experiments.daisy_cable_replanning.inverse_kinematics import (
    IKSolverException,
    InverseKinematicsSolver,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.joint_monitors import (
    CloseToJointConfigurations,
    JointConfigurationSet,
    RecentJointStates,
)
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointWaypointPath,
    RetraceRecentJointStates,
    RetractToLastSafeJointConfiguration,
)
from krrood.adapters.json_serializer import to_json
from krrood_spatial_math.spatial_types import Pose
from ompl import base as ompl_base
from ompl import geometric as ompl_geometric
from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
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

    clearance: float = 0.2
    """
    Joint distance, in radians, the planned path keeps from every illegal joint state.
    """

    collision_margin: float = 0.001
    """
    Distance in metres below which a body of the planned arm counts as colliding.
    """

    validity_checking_resolution: float = 0.002
    """
    Spacing at which the planner checks a path segment, as a fraction of the extent of
    the joint space.
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

    goal_seeds: int = 64
    """
    How many joint states, besides the current one, inverse kinematics starts from to
    find alternative goal configurations.
    """

    goal_seed_spread: float = pi
    """
    Largest offset of a joint of a seed from its current position, in radians.
    """

    goal_sampling_seed: int = 0
    """
    Seed of the random offsets of the seeds, so a replanning is reproducible.
    """

    stagnation_timeout: float = 1.5
    """
    Simulated time, in seconds, a motion may go without approaching its target before
    it gives way to a planned path, when stagnation triggers replanning.
    """

    retract_clearance: float = 0.0
    """
    Distance in metres, on top of the buffer zone of each pair of bodies, that the arm's
    bodies keep in the joint state a stuck motion retracts to, when it retracts first.
    """

    retraced_control_cycles: int = 200
    """
    How many of its latest control cycles a stuck arm retraces, when it retracts to
    :attr:`RetractTarget.RECENT_STEPS`.
    """

    distinct_goal_distance: float = 0.05
    """
    Largest joint difference, in radians, below which two goal configurations count as
    the same.
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
        self.write(self._original_positions)

    def is_collision_free(
        self, positions: np.ndarray, collision_matrix: CollisionMatrix
    ) -> bool:
        """
        :param positions: Positions of :attr:`connections` to probe.
        :param collision_matrix: The pairs that may not collide, each with the distance
            below which it counts as colliding.
        :return: Whether no pair is closer than its distance at the positions.
        """
        return not self.contacts_closer_than_allowed(positions, collision_matrix)

    def contacts_closer_than_allowed(
        self, positions: np.ndarray, collision_matrix: CollisionMatrix
    ) -> list[ClosestPoints]:
        """
        :param positions: Positions of :attr:`connections` to probe.
        :param collision_matrix: The pairs that may not collide, each with the distance
            below which it counts as colliding.
        :return: The pairs closer than their distance at the positions.
        """
        self.write(positions)
        # The detector may also report pairs that are a little further apart than
        # their check's distance, so the distance is compared here.
        allowed_distances = {
            frozenset((check.body_a, check.body_b)): check.distance
            for check in collision_matrix.collision_checks
        }
        collision_detector = self.world.collision_manager.collision_detector
        return [
            contact
            for contact in collision_detector.check_collisions(
                collision_matrix
            ).contacts
            if contact.distance
            < allowed_distances[frozenset((contact.body_a, contact.body_b))]
        ]

    def write(self, positions: np.ndarray) -> None:
        """
        Write the positions of :attr:`connections` and bring the collision detector up
        to date with them.
        """
        self.world.state.positions[self._columns] = positions
        self.world.collision_manager.collision_detector.world_state_updater.on_state_change()


# %% goal configurations


@dataclass
class GoalConfigurationSampler:
    """
    Finds several joint configurations of an arm that put its tool frame at a pose, by
    starting inverse kinematics from the current joint state and from randomly offset
    ones.
    """

    world: World
    """
    The world the arm is in.
    """

    arm: Arm
    """
    The arm whose joint configurations are sought.
    """

    root: Body
    """
    The body the arm is mounted below.
    """

    parameters: ReplanningParameters
    """
    How many seeds are tried and how far they spread.
    """

    def configurations_reaching(self, pose: Pose) -> list[np.ndarray]:
        """
        :param pose: Where the tool frame is to be.
        :return: The distinct configurations of the arm's joints that inverse
            kinematics found from the seeds, the one from the current joint state
            first; empty if it found none.
        """
        connections = self.arm.active_connections
        configurations: list[np.ndarray] = []
        with ProbedJointState(world=self.world, connections=connections) as probe:
            for seed in self._seeds(connections):
                probe.write(seed)
                configuration = self._solve_from_current_state(pose)
                if configuration is not None and self._is_distinct(
                    configuration, configurations
                ):
                    configurations.append(configuration)
        return configurations

    def _solve_from_current_state(self, pose: Pose) -> np.ndarray | None:
        """
        :return: The configuration inverse kinematics converges to from the joint state
            the world is in, or ``None`` if it does not converge to the pose.
        """
        # Approved exception to the no-try-except rule: the solver reports a seed that
        # does not converge only by raising, and other seeds may still succeed.
        try:
            solution = InverseKinematicsSolver(self.world).solve(
                root=self.root,
                tip=self.arm.end_effector.tool_frame,
                target=pose.homogeneous_matrix,
            )
        except IKSolverException:
            return None
        return np.array(
            [solution[connection.dof] for connection in self.arm.active_connections]
        )

    def _seeds(self, connections: list[ActiveConnection1DOF]) -> list[np.ndarray]:
        """
        :return: The current joint state, followed by randomly offset ones within the
            joint limits.
        """
        current = np.array([connection.position for connection in connections])
        lower = np.array(
            [connection.dof.limits.lower.position for connection in connections]
        )
        upper = np.array(
            [connection.dof.limits.upper.position for connection in connections]
        )
        random_generator = np.random.default_rng(self.parameters.goal_sampling_seed)
        offsets = random_generator.uniform(
            -self.parameters.goal_seed_spread,
            self.parameters.goal_seed_spread,
            size=(self.parameters.goal_seeds, len(connections)),
        )
        return [current, *np.clip(current + offsets, lower, upper)]

    def _is_distinct(
        self, configuration: np.ndarray, configurations: list[np.ndarray]
    ) -> bool:
        """
        :return: Whether the configuration differs from every one found before.
        """
        return all(
            np.max(np.abs(configuration - other))
            > self.parameters.distinct_goal_distance
            for other in configurations
        )


# %% planning


@dataclass
class JointSpacePlanner:
    """
    Plans paths of the joints of one arm with OMPL's RRT-Connect that are free of
    collisions and keep clear of a set of illegal joint configurations, with every other
    joint held where it is.
    """

    world: World
    """
    The world the arm is in.
    """

    robot: DAiSy
    """
    The robot the arm belongs to.
    """

    arm: Arm
    """
    The arm whose joints are planned.
    """

    ignored_bodies: set[Body]
    """
    Bodies whose collisions do not count, such as a soft cable that the arm is meant to
    touch.
    """

    illegal_configurations: JointConfigurationSet
    """
    The joint configurations the path keeps clear of.
    """

    parameters: ReplanningParameters
    """
    How the path is planned.
    """

    def collision_matrix(self) -> CollisionMatrix:
        """
        :return: The pairs of bodies, apart from :attr:`ignored_bodies`, the arm may
            not bring closer than the collision margin, as the world is attached when
            this is asked.
        """
        collision_matrix = ArmCollisionPairs(
            world=self.world, robot=self.robot
        ).collision_matrix_of(self.arm, self.parameters.collision_margin)
        return CollisionMatrix(
            {
                check
                for check in collision_matrix.collision_checks
                if not {check.body_a, check.body_b} & self.ignored_bodies
            }
        )

    def plan(self, goals: list[np.ndarray]) -> list[np.ndarray]:
        """
        :param goals: Alternative positions of the arm's joints at the goal.
        :return: Waypoints from their current positions to the nearest goal a path is
            found to, no further apart than
            :attr:`ReplanningParameters.waypoint_spacing`.
        :raises NoValidGoalConfiguration: If every goal is in collision or too close to
            an illegal configuration.
        :raises NoJointPathFound: If no path to any valid goal is found.
        """
        connections = self.arm.active_connections
        collision_matrix = self.collision_matrix()
        start = np.array([connection.position for connection in connections])
        held_configuration = self.illegal_configurations.current_positions()
        planned_columns = [
            self.illegal_configurations.connections.index(connection)
            for connection in connections
        ]
        space = self._state_space(connections)
        with ProbedJointState(world=self.world, connections=connections) as probe:

            def distance_to_illegal_configurations(positions: np.ndarray) -> float:
                configuration = held_configuration.copy()
                configuration[planned_columns] = positions
                return self.illegal_configurations.distance_to(configuration)

            def is_valid(positions: np.ndarray) -> bool:
                return distance_to_illegal_configurations(
                    positions
                ) >= self.parameters.clearance and probe.is_collision_free(
                    positions, collision_matrix
                )

            # The nearest goal keeps the arm close to the joint states the plan expects.
            reachable_goals = sorted(
                (goal for goal in goals if is_valid(goal)),
                key=lambda goal: np.max(np.abs(goal - start)),
            )
            if not reachable_goals:
                raise NoValidGoalConfiguration(
                    start=start.tolist(),
                    rejected_goals=[
                        RejectedGoalConfiguration(
                            positions=goal.tolist(),
                            distance_to_illegal_configurations=distance_to_illegal_configurations(
                                goal
                            ),
                            contacts=probe.contacts_closer_than_allowed(
                                goal, collision_matrix
                            ),
                        )
                        for goal in goals
                    ],
                    clearance=self.parameters.clearance,
                )
            for goal in reachable_goals:
                waypoints = self._solve(space, is_valid, start, goal)
                if waypoints is not None:
                    return waypoints
        raise NoJointPathFound(
            start=start.tolist(),
            goals=[goal.tolist() for goal in reachable_goals],
            planning_time=self.parameters.planning_time,
        )

    def _solve(
        self,
        space: ompl_base.RealVectorStateSpace,
        is_valid: Callable[[np.ndarray], bool],
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray] | None:
        """
        :return: The waypoints of the simplified path RRT-Connect found from the start
            to the goal, or ``None`` if it found none within the planning time.
        """
        setup = ompl_geometric.SimpleSetup(space)
        setup.setStateValidityChecker(
            lambda state: is_valid(
                np.array([state[index] for index in range(len(start))])
            )
        )
        setup.setStartAndGoalStates(
            self._ompl_state(space, start), self._ompl_state(space, goal)
        )
        setup.setPlanner(ompl_geometric.RRTConnect(setup.getSpaceInformation()))
        setup.getSpaceInformation().setStateValidityCheckingResolution(
            self.parameters.validity_checking_resolution
        )
        solved = setup.solve(self.parameters.planning_time)
        if not (solved and setup.haveExactSolutionPath()):
            return None
        # Without a duration, the simplification does a fixed amount of work, so the
        # path does not depend on how fast the machine is.
        setup.simplifySolution()
        return self._waypoints_along(setup.getSolutionPath(), len(start))

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


@dataclass(eq=False, repr=False, kw_only=True)
class PlannedJointPathToPose(JointWaypointPath):
    """
    Moves an arm along a path planned, once it starts, from where the arm is to a joint
    configuration that puts its tool frame at a goal pose.
    """

    planner: JointSpacePlanner
    """
    Plans the path.
    """

    goal_sampler: GoalConfigurationSampler
    """
    Finds the joint configurations that put the arm's tool frame at the goal pose.
    """

    goal_pose: Pose
    """
    Where the tool frame is brought.
    """

    replanning_log: ReplanningLog
    """
    Where the planned path is recorded.
    """

    def plan_waypoints(self) -> list[np.ndarray]:
        goals = self.goal_sampler.configurations_reaching(self.goal_pose)
        waypoints = self.planner.plan(goals)
        self.replanning_log.record(waypoints)
        return waypoints


# %% retracting


class RetractTarget(StrEnum):
    """
    Where a stuck arm retracts to, along the path it came, before the path to the
    motion's target is planned.
    """

    LAST_CLEAR_STATE = "last-clear-state"
    """
    The last joint state of the motion in which every pair of the arm's bodies kept its
    Giskard buffer zone plus :attr:`ReplanningParameters.retract_clearance`, or the
    motion's start if there was none.
    """

    MOTION_START = "motion-start"
    """
    The joint state the motion started at, where the previous motion reached its goal.
    """

    RECENT_STEPS = "recent-steps"
    """
    The joint state the motion held :attr:`ReplanningParameters.retraced_control_cycles`
    control cycles ago, or its start if it ran shorter; the arm retraces every control
    cycle in between with Giskard's collision avoidance of the arm off.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class PathSinceMotionStart(EndedByOwner, MotionStatechartNode):
    """
    Remembers the joint states an arm passed since it started; observes False
    throughout.
    """

    connections: list[ActiveConnection1DOF]
    """
    The joints of the arm.
    """

    waypoint_spacing: float
    """
    Largest joint difference, in radians, between consecutive remembered joint states.
    """

    passed_positions: list[np.ndarray] = field(init=False, default_factory=list)
    """
    The remembered positions of :attr:`connections`, the earliest first.
    """

    def on_start(self) -> None:
        self.passed_positions = [self._current_positions()]

    def on_tick(self) -> ObservationStateValue:
        positions = self._current_positions()
        if (
            np.max(np.abs(positions - self.passed_positions[-1]))
            >= self.waypoint_spacing
        ):
            self.passed_positions.append(positions)
        return ObservationStateValue.FALSE

    def _current_positions(self) -> np.ndarray:
        """
        :return: The positions :attr:`connections` are at.
        """
        return np.array([connection.position for connection in self.connections])


@dataclass(eq=False, repr=False, kw_only=True)
class PathSinceLastClearJointState(PathSinceMotionStart):
    """
    Remembers the joint states an arm passed since its last clear joint state, the one
    at which every pair of its bodies kept at least the pair's clearance, or since it
    started if there was none.
    """

    clearances: CollisionMatrix
    """
    The pairs of bodies, each with the distance it has to keep for a joint state to
    count as clear.
    """

    def on_tick(self) -> ObservationStateValue:
        positions = self._current_positions()
        if not ProbedJointState(
            world=self.world, connections=self.connections
        ).is_collision_free(positions, self.clearances):
            return super().on_tick()
        self.passed_positions = [positions]
        return ObservationStateValue.FALSE


@dataclass(eq=False, repr=False, kw_only=True)
class RetractAlongRememberedPath(JointWaypointPath):
    """
    Moves the joints a :class:`PathSinceMotionStart` watches back along the remembered
    path to where it begins.
    """

    monitor: PathSinceMotionStart
    """
    The node that remembers the path.
    """

    connections: list[ActiveConnection1DOF] = field(init=False)
    """
    The joints the monitor watches.
    """

    def __post_init__(self):
        self.connections = self.monitor.connections
        super().__post_init__()

    def plan_waypoints(self) -> list[np.ndarray]:
        return list(reversed(self.monitor.passed_positions))


# %% recovering a stuck motion


@dataclass
class RecoveryBeforeStuckMotion(SequentialInsertion[StoppedWhenTrue]):
    """
    Inserts the steps recovering a motion its monitor stopped right before the step
    holding it, so the motion runs again from where they leave the arm.
    """

    steps: list[StatechartNode]
    """
    The steps recovering the motion, in the order they run.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: StoppedWhenTrue) -> bool:
        return True

    def anchor(self, plan_node: StoppedWhenTrue) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: StoppedWhenTrue) -> list[StatechartNode]:
        return [Sequence(name="recovery of the stuck motion", nodes=self.steps)]


# %% plan transformations


@dataclass
class MovingArm:
    """
    An arm whose path is planned, with the robot and the world it moves in.
    """

    arm: Arm
    """
    The arm whose path is planned.
    """

    robot: DAiSy
    """
    The robot the arm belongs to.
    """

    world: World
    """
    The world the robot is in.
    """

    @staticmethod
    def arm_with_tool_frame(
        robot: DAiSy, tool_frame: KinematicStructureEntity
    ) -> Arm | None:
        """
        :return: The arm of the robot whose tool frame is `tool_frame`, if any.
        """
        return next(
            (
                arm
                for arm in robot.all_arms
                if arm.end_effector.tool_frame is tool_frame
            ),
            None,
        )

    def path_planner(
        self,
        parameters: ReplanningParameters,
        illegal_configurations: JointConfigurationSet,
    ) -> JointSpacePlanner:
        """
        :param parameters: How the path is planned.
        :param illegal_configurations: The joint configurations the path keeps clear
            of.
        :return: The planner of the arm's path, ignoring the soft cable.
        """
        return JointSpacePlanner(
            world=self.world,
            robot=self.robot,
            arm=self.arm,
            # The cable is soft and meant to be touched; the demo allows its collisions.
            ignored_bodies={
                cable.root
                for cable in self.world.get_semantic_annotations_by_type(Cable)
            },
            illegal_configurations=illegal_configurations,
            parameters=parameters,
        )

    def goal_sampler(
        self, parameters: ReplanningParameters
    ) -> GoalConfigurationSampler:
        """
        :param parameters: How many seeds are tried and how far they spread.
        :return: The sampler of the arm's joint configurations reaching a pose.
        """
        return GoalConfigurationSampler(
            world=self.world, arm=self.arm, root=self.robot.root, parameters=parameters
        )

    def without_collision_avoidance(self, path: JointWaypointPath) -> Parallel:
        """
        :param path: The node moving the arm along a path checked against the bodies
            themselves, such as a planned or an already driven one.
        :return: The path, followed while Giskard's collision avoidance leaves the
            arm's bodies alone, since it would keep the arm out of the buffer zones the
            path may pass through.
        """
        arm_bodies = ArmCollisionPairs(
            world=self.world, robot=self.robot
        ).bodies_of_arm[self.arm]
        return Parallel(
            name=f"{path.name} without collision avoidance of the arm",
            nodes=[
                UpdateTemporaryCollisionRules(
                    name="allow collisions of the arm",
                    temporary_rules=[
                        AllowCollisionForBodies(allowed_collision_bodies=arm_bodies)
                    ],
                ),
                path,
            ],
        )


@dataclass
class ReplannedGoal(MovingArm):
    """
    A pose goal of an arm's tool frame that is replaced, with what replanning it reads.
    """

    goal: CartesianPose
    """
    A copy of the goal, not yet part of a statechart.
    """


@dataclass
class MonitoredGoalWithRecovery:
    """
    A pose goal its monitor stops, and the steps run before it runs again.
    """

    monitored_goal: StoppedWhenTrue
    """
    The goal, stopped once the monitor fires.
    """

    recovery_steps: list[StatechartNode]
    """
    The steps run once the monitor stopped the goal, in the order they run.
    """


@dataclass
class ReplanningTransformation(PlanTransformation[CartesianPose], ABC):
    """
    Replaces the pose goals of an arm's tool frame that allow replanning and are steps of
    a sequence by its subclass's :meth:`replacement_for`, which reaches the goal pose
    along a planned joint path once the goal gets stuck.
    """

    parameters: ReplanningParameters = field(kw_only=True)
    """
    When a goal gives way and how the path is planned.
    """

    replanning_log: ReplanningLog = field(kw_only=True)
    """
    Where the planned paths are recorded.
    """

    def is_applicable(self, plan_node: CartesianPose) -> bool:
        """
        A goal is replanned in a :class:`~cramph.composites.Sequence`, which runs the
        recovery as a step before it.
        """
        return (
            plan_node.allow_replanning
            and isinstance(plan_node.parent_node, Sequence)
            and self._arm_moved_by(plan_node) is not None
        )

    def apply(self, plan_node: CartesianPose) -> None:
        robot = plan_node.context.require_extension(extension_type=RobotAccess).robot
        plan_node.statechart.replace(
            node=plan_node,
            replacement=self.replacement_for(
                ReplannedGoal(
                    goal=self._copy_of(plan_node),
                    arm=self._arm_moved_by(plan_node),
                    robot=robot,
                    world=plan_node.world,
                )
            ),
        )

    @abstractmethod
    def replacement_for(self, replanned: ReplannedGoal) -> StatechartNode:
        """
        :param replanned: The pose goal to replace.
        :return: The node running in the goal's place.
        """

    @staticmethod
    def _arm_moved_by(goal: CartesianPose) -> Arm | None:
        """
        :return: The arm of the robot whose tool frame the goal moves, if any.
        """
        robot = goal.context.require_extension(extension_type=RobotAccess).robot
        return MovingArm.arm_with_tool_frame(robot, goal.tip_link)

    @staticmethod
    def _copy_of(goal: CartesianPose) -> CartesianPose:
        """
        :return: A new goal reaching the same pose, handed copies of the nodes the goal
            was handed, which the plan's rewriting leaves alone.
        """
        copy = replace(goal)
        for handed_node in goal.nodes[len(goal.create_tasks()) :]:
            copy.add_node(node=replace(handed_node))
        goal.context.require_extension(extension_type=PlanRewriting).treat_as_offered(
            copy
        )
        return copy

    def _stalled(self, monitored_node: StatechartNode) -> Stalled:
        """
        :return: The monitor turning True once the node stops approaching its goal.
        """
        return Stalled(
            name="no longer progressing",
            monitored_node=monitored_node,
            timeout=timedelta(seconds=self.parameters.stagnation_timeout),
        )


@dataclass
class InsertedRecoveryReplanning(ReplanningTransformation, ABC):
    """
    Puts each goal under failure handling that, once its subclass's monitor stopped the
    goal, inserts the steps recovering it before the goal and runs the goal again.

    .. warning:: The recovery changes the statechart while it runs, so it only works in a
        statechart ticked in this process.
    """

    def replacement_for(self, replanned: ReplannedGoal) -> HandledOnFailure:
        monitored = self.monitored_goal_with_recovery(replanned)
        return HandledOnFailure(
            name=replanned.goal.name,
            handled_node=monitored.monitored_goal,
            failure_handling_strategy=RecoveryByInsertion(
                maximum_attempts=1,
                transformation=RecoveryBeforeStuckMotion(
                    steps=monitored.recovery_steps
                ),
            ),
        )

    @abstractmethod
    def monitored_goal_with_recovery(
        self, replanned: ReplannedGoal
    ) -> MonitoredGoalWithRecovery:
        """
        :param replanned: The pose goal to replace.
        :return: The goal, stopped once this transformation's monitor fires, with the
            steps recovering it.
        """

    def _planned_path(
        self,
        replanned: ReplannedGoal,
        illegal_configurations: JointConfigurationSet,
    ) -> PlannedJointPathToPose:
        """
        :param illegal_configurations: The joint configurations the path keeps clear
            of.
        :return: The path of the arm to the goal pose, planned once it starts.
        """
        return PlannedJointPathToPose(
            name="planned joint path",
            connections=replanned.arm.active_connections,
            planner=replanned.path_planner(self.parameters, illegal_configurations),
            goal_sampler=replanned.goal_sampler(self.parameters),
            goal_pose=replanned.goal.goal_pose,
            replanning_log=self.replanning_log,
        )


@dataclass
class IllegalJointStateReplanning(InsertedRecoveryReplanning):
    """
    Stops the pose goals of the actions illegal joint states were recorded in as soon as
    the arms come close to one, retracts to the last safe joint state and plans a path
    to the goal pose around the illegal joint states.
    """

    illegal_joint_states: IllegalJointStateDatabase = field(kw_only=True)
    """
    The joint states to stay away from.
    """

    def is_applicable(self, plan_node: CartesianPose) -> bool:
        return super().is_applicable(plan_node) and type(
            plan_node.parent_node
        ).__name__ in {
            state.motion.action
            for state in self.illegal_joint_states.illegal_joint_states
        }

    def monitored_goal_with_recovery(
        self, replanned: ReplannedGoal
    ) -> MonitoredGoalWithRecovery:
        illegal_configurations = self.illegal_joint_states.joint_configurations(
            replanned.world
        )
        monitor = CloseToJointConfigurations(
            name="close to illegal joint state",
            configurations=illegal_configurations,
            threshold=self.parameters.trigger_distance,
            safe_distance=self.parameters.safe_distance,
        )
        return MonitoredGoalWithRecovery(
            monitored_goal=StoppedWhenTrue(
                monitor=monitor, monitored_node=replanned.goal
            ),
            recovery_steps=[
                RetractToLastSafeJointConfiguration(
                    name="retract to last safe joint state", monitor=monitor
                ),
                self._planned_path(replanned, illegal_configurations),
            ],
        )


@dataclass
class StagnationReplanning(InsertedRecoveryReplanning):
    """
    Stops every pose goal once it stops approaching its pose short of reaching it, as it
    does when collision avoidance holds the arm back, and plans a collision-free path to
    the pose, from where the arm got stuck or after retracting it to
    :attr:`retract_target`.
    """

    retract_target: RetractTarget | None = field(default=None, kw_only=True)
    """
    Where a stuck arm retracts to before the path is planned, or ``None`` to plan from
    where it got stuck.
    """

    def monitored_goal_with_recovery(
        self, replanned: ReplannedGoal
    ) -> MonitoredGoalWithRecovery:
        path = self._planned_path(
            replanned,
            IllegalJointStateDatabase.for_daisy_arms().joint_configurations(
                replanned.world
            ),
        )
        planned_path = replanned.without_collision_avoidance(path)
        goal = replanned.goal
        if self.retract_target is None:
            return MonitoredGoalWithRecovery(
                monitored_goal=StoppedWhenTrue(
                    monitor=self._stalled(goal), monitored_node=goal
                ),
                recovery_steps=[planned_path],
            )
        retract = self._retract(replanned, path.planner)
        goal_remembering_its_path = Parallel(
            name=f"{goal.name} remembering its path",
            nodes=[goal, retract.monitor],
            minimum_success=1,
        )
        return MonitoredGoalWithRecovery(
            monitored_goal=StoppedWhenTrue(
                monitor=self._stalled(goal_remembering_its_path),
                monitored_node=goal_remembering_its_path,
            ),
            recovery_steps=[
                (
                    replanned.without_collision_avoidance(retract)
                    if self.retract_target is RetractTarget.RECENT_STEPS
                    else retract
                ),
                planned_path,
            ],
        )

    def _retract(
        self, replanned: ReplannedGoal, planner: JointSpacePlanner
    ) -> RetractAlongRememberedPath | RetraceRecentJointStates:
        """
        :param planner: The planner of the path planned after retracting, whose body
            pairs decide whether a joint state is clear.
        :return: The node moving the arm back to :attr:`retract_target`, whose monitor
            remembers the way back while the goal runs.
        """
        name = f"retract to {self.retract_target}"
        connections = replanned.arm.active_connections
        match self.retract_target:
            case RetractTarget.RECENT_STEPS:
                return RetraceRecentJointStates(
                    name=name,
                    monitor=RecentJointStates(
                        name="recent joint states",
                        connections=connections,
                        control_cycles=self.parameters.retraced_control_cycles,
                    ),
                )
            case RetractTarget.MOTION_START:
                return RetractAlongRememberedPath(
                    name=name,
                    monitor=PathSinceMotionStart(
                        name="path since motion start",
                        connections=connections,
                        waypoint_spacing=self.parameters.waypoint_spacing,
                    ),
                )
            case RetractTarget.LAST_CLEAR_STATE:
                return RetractAlongRememberedPath(
                    name=name,
                    monitor=PathSinceLastClearJointState(
                        name="path since last clear joint state",
                        connections=connections,
                        waypoint_spacing=self.parameters.waypoint_spacing,
                        clearances=self._retract_clearances(replanned, planner),
                    ),
                )

    def _retract_clearances(
        self, replanned: ReplannedGoal, planner: JointSpacePlanner
    ) -> CollisionMatrix:
        """
        :return: The pairs of bodies the planner checks the path against, each with its
            Giskard buffer zone plus :attr:`ReplanningParameters.retract_clearance`.
        """
        manager = replanned.world.collision_manager
        return CollisionMatrix(
            {
                CollisionCheck(
                    check.body_a,
                    check.body_b,
                    manager.get_buffer_zone_distance(check.body_a, check.body_b)
                    + self.parameters.retract_clearance,
                )
                for check in planner.collision_matrix().collision_checks
            }
        )
