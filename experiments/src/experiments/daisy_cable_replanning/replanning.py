from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import timedelta
from itertools import pairwise
from math import ceil, pi
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
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.joint_monitors import (
    CloseToJointConfigurations,
    JointConfigurationSet,
)
from giskardpy.motion_statechart.monitors.progress_monitors import NoLongerProgressing
from giskardpy.motion_statechart.monitors.templates import ReplacedWhenTrue
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointWaypointPath,
    RetractToLastSafeJointConfiguration,
)
from krrood.adapters.json_serializer import to_json
from ompl import base as ompl_base
from ompl import geometric as ompl_geometric
from semantic_digital_twin.collision_checking.collision_matrix import CollisionMatrix
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_computations.ik_solver import IKSolverException
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

    clearance: float = 0.2
    """
    Joint distance, in radians, the planned path keeps from every illegal joint state.
    """

    collision_margin: float = 0.002
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
        self.write(positions)
        # The detector may also report pairs that are a little further apart than
        # their check's distance, so the distance is compared here.
        allowed_distances = {
            frozenset((check.body_a, check.body_b)): check.distance
            for check in collision_matrix.collision_checks
        }
        collision_detector = self.world.collision_manager.collision_detector
        return not any(
            contact.distance
            < allowed_distances[frozenset((contact.body_a, contact.body_b))]
            for contact in collision_detector.check_collisions(
                collision_matrix
            ).contacts
        )

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
            solution = self.world.compute_inverse_kinematics(
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
        :raises NoJointPathFound: If no path to any goal is found.
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

            def is_valid(positions: np.ndarray) -> bool:
                configuration = held_configuration.copy()
                configuration[planned_columns] = positions
                return self.illegal_configurations.distance_to(
                    configuration
                ) >= self.parameters.clearance and probe.is_collision_free(
                    positions, collision_matrix
                )

            # The nearest goal keeps the arm close to the joint states the plan expects.
            reachable_goals = sorted(
                (goal for goal in goals if is_valid(goal)),
                key=lambda goal: np.max(np.abs(goal - start)),
            )
            for goal in reachable_goals:
                waypoints = self._solve(space, is_valid, start, goal)
                if waypoints is not None:
                    return waypoints
        raise NoJointPathFound(
            start=start.tolist(),
            goals=[goal.tolist() for goal in goals],
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

    goal_sampler: GoalConfigurationSampler = field(kw_only=True)
    """
    Finds the joint configurations that put the arm's tool frame at the goal pose.
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
        goals = self.goal_sampler.configurations_reaching(self.goal_pose)
        waypoints = self.planner.plan(goals)
        self.replanning_log.record(waypoints)
        return waypoints


# %% replanned motions


@dataclass
class ToolCenterPointMotionWithReplanning(MoveToolCenterPointMotion):
    """
    Moves the tool center point like :class:`MoveToolCenterPointMotion`, but gives way
    to a joint path planned to the same target when its subclass's trigger fires.
    """

    parameters: ReplanningParameters = field(kw_only=True)
    """
    When the motion gives way and how the replacing path is planned.
    """

    replanning_log: ReplanningLog = field(kw_only=True)
    """
    Where the replacing paths are recorded.
    """

    @classmethod
    def replacing(cls, motion: MoveToolCenterPointMotion, **replanning: object) -> Self:
        """
        :param motion: The motion to replace.
        :param replanning: The fields this class adds to the motion's own.
        :return: A motion of this class with the parameters of the given motion.
        """
        values = vars(motion)
        return cls(
            **{
                motion_field.name: values[motion_field.name]
                for motion_field in fields(motion)
                if motion_field.init
            },
            **replanning,
        )

    def _planner(
        self, illegal_configurations: JointConfigurationSet
    ) -> JointSpacePlanner:
        """
        :param illegal_configurations: The joint configurations the path keeps clear
            of.
        :return: The planner of the arm's path, ignoring the soft cable.
        """
        return JointSpacePlanner(
            world=self.world,
            robot=self.robot,
            arm=self.arm,
            # The cable is soft and meant to be touched; the demo allows its
            # collisions.
            ignored_bodies={
                cable.root
                for cable in self.world.get_semantic_annotations_by_type(Cable)
            },
            illegal_configurations=illegal_configurations,
            parameters=self.parameters,
        )

    @property
    def _goal_sampler(self) -> GoalConfigurationSampler:
        """
        :return: The sampler of the arm's joint configurations reaching the target.
        """
        return GoalConfigurationSampler(
            world=self.world,
            arm=self.arm,
            root=self.robot.root,
            parameters=self.parameters,
        )

    def _planned_path(
        self, illegal_configurations: JointConfigurationSet
    ) -> PlannedJointPathToPose:
        """
        :param illegal_configurations: The joint configurations the path keeps clear
            of.
        :return: The path to the target, planned once it starts.
        """
        return PlannedJointPathToPose(
            name="planned joint path",
            connections=self.arm.active_connections,
            planner=self._planner(illegal_configurations),
            goal_sampler=self._goal_sampler,
            goal_pose=self.target,
            replanning_log=self.replanning_log,
        )

    def _without_collision_avoidance_of_arm(
        self, path: MotionStatechartNode
    ) -> Sequence:
        """
        :param path: The node moving the arm along a planned path.
        :return: The path, followed while Giskard's collision avoidance leaves the
            arm's bodies alone, and the collision rules restored after it.

        The planner has checked the path against the bodies themselves; Giskard's
        avoidance would keep the arm out of its buffer zones around them, which the
        path may pass through, and hold it back again.
        """
        arm_bodies = ArmCollisionPairs(
            world=self.world, robot=self.robot
        ).bodies_of_arm[self.arm]
        return Sequence(
            name="follow planned path",
            nodes=[
                Parallel(
                    name="planned path without collision avoidance of the arm",
                    nodes=[
                        UpdateTemporaryCollisionRules(
                            name="allow collisions of the arm",
                            temporary_rules=[
                                AllowCollisionForBodies(
                                    allowed_collision_bodies=arm_bodies
                                )
                            ],
                        ),
                        path,
                    ],
                ),
                UpdateTemporaryCollisionRules(
                    name="restore collision rules",
                    temporary_rules=list(self.world.collision_manager.temporary_rules),
                ),
            ],
        )


@dataclass
class ReplannedToolCenterPointMotion(ToolCenterPointMotionWithReplanning):
    """
    Leaves the motion as soon as the arms come close to an illegal joint state,
    retracts to the last safe joint state and reaches the target along a path planned
    around the illegal joint states.
    """

    illegal_joint_states: IllegalJointStateDatabase = field(kw_only=True)
    """
    The joint states to stay away from.
    """

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
                    self._planned_path(illegal_configurations),
                ],
            ),
        )


@dataclass
class StagnationReplannedToolCenterPointMotion(ToolCenterPointMotionWithReplanning):
    """
    Leaves the motion once it stops approaching its target short of reaching it, as it
    does when collision avoidance holds the arm back, and reaches the target from where
    the arm got stuck along a collision-free planned path.
    """

    @property
    def _motion_chart(self) -> ReplacedWhenTrue:
        motion = super()._motion_chart
        return ReplacedWhenTrue(
            name="MoveTCP or replanned path",
            monitor=NoLongerProgressing(
                name="no longer progressing",
                monitored_node=motion,
                timeout=timedelta(seconds=self.parameters.stagnation_timeout),
            ),
            monitored_node=motion,
            replacement=self._without_collision_avoidance_of_arm(
                self._planned_path(
                    IllegalJointStateDatabase.for_daisy_arms().joint_configurations(
                        self.world
                    )
                )
            ),
        )


# %% plan transformations


@dataclass
class ReplanningTransformation(PlanTransformation[ActionDescription], ABC):
    """
    Replaces the tool center point motions of the actions it applies to by motions
    that give way to a planned path.
    """

    parameters: ReplanningParameters = field(kw_only=True)
    """
    When a motion gives way and how the replacing path is planned.
    """

    replanning_log: ReplanningLog = field(kw_only=True)
    """
    Where the replacing paths are recorded.
    """

    @abstractmethod
    def replacement_for(
        self, motion: MoveToolCenterPointMotion
    ) -> ToolCenterPointMotionWithReplanning:
        """
        :return: The motion replacing the given one.
        """

    def apply(self, plan_node: ActionNode) -> None:
        for node in plan_node.descendants:
            if not isinstance(node, MotionNode) or type(node.designator) is not (
                MoveToolCenterPointMotion
            ):
                continue
            self.rewrite_motion_node(node)

    def rewrite_motion_node(self, node: MotionNode) -> None:
        """
        Replace the tool center point motion of the node by :meth:`replacement_for`.

        :param node: A node of a :class:`MoveToolCenterPointMotion`.
        """
        node.designator = self.replacement_for(node.designator)
        node.designator.plan_node = node


@dataclass
class IllegalJointStateReplanning(ReplanningTransformation):
    """
    Replaces the tool center point motions of the actions illegal joint states were
    recorded in by :class:`ReplannedToolCenterPointMotion`.
    """

    illegal_joint_states: IllegalJointStateDatabase = field(kw_only=True)
    """
    The joint states to stay away from.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return type(plan_node.action).__name__ in {
            state.motion.action
            for state in self.illegal_joint_states.illegal_joint_states
        }

    def replacement_for(
        self, motion: MoveToolCenterPointMotion
    ) -> ReplannedToolCenterPointMotion:
        return ReplannedToolCenterPointMotion.replacing(
            motion,
            illegal_joint_states=self.illegal_joint_states,
            parameters=self.parameters,
            replanning_log=self.replanning_log,
        )


@dataclass
class StagnationReplanning(ReplanningTransformation):
    """
    Replaces the tool center point motions of every action by
    :class:`StagnationReplannedToolCenterPointMotion`.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return True

    def replacement_for(
        self, motion: MoveToolCenterPointMotion
    ) -> StagnationReplannedToolCenterPointMotion:
        return StagnationReplannedToolCenterPointMotion.replacing(
            motion, parameters=self.parameters, replanning_log=self.replanning_log
        )
