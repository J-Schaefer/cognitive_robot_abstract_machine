from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode, PlanNode
from giskardpy.motion_statechart.monitors.joint_monitors import JointConfigurationSet
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyJoint
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Self

# %% illegal joint states


class CollisionKind(StrEnum):
    """
    Which parts of the robot touch each other in a collision.
    """

    BETWEEN_ARMS = "between_arms"
    """
    A body of the left arm touches a body of the right arm.
    """

    WITHIN_ARM = "within_arm"
    """
    Two bodies of the same arm touch each other.
    """

    ARM_AND_ROBOT_BODY = "arm_and_robot_body"
    """
    A body of an arm touches a body of the robot that belongs to neither arm.
    """


@dataclass(frozen=True)
class MotionLabel:
    """
    Identifies a motion of the cable demo across runs of it.
    """

    stage: str
    """
    The stage of the demo the motion is performed in.
    """

    index: int
    """
    How many motions of the stage started before this one.
    """

    action: str
    """
    Name of the type of the innermost action the motion belongs to, or ``plan`` if it
    belongs to none.
    """

    motion: str
    """
    Name of the type of the motion.
    """

    def __str__(self) -> str:
        return f"{self.stage}#{self.index} {self.action}.{self.motion}"


@dataclass
class IllegalJointState:
    """
    Joint positions of both arms in which the robot collides with itself.
    """

    positions: list[float]
    """
    Positions of the arm joints, ordered as
    :attr:`IllegalJointStateDatabase.joint_names`.
    """

    colliding_joint_names: list[str]
    """
    Names of the joints of the arms whose bodies collide; only their positions decide
    whether a joint state is close to this one.
    """

    collision_kind: CollisionKind
    """
    Which parts of the robot collide.
    """

    first_body: str
    """
    Name of one of the colliding bodies.
    """

    second_body: str
    """
    Name of the other colliding body.
    """

    penetration_depth: float
    """
    How deep the bodies reach into each other, in metres.
    """

    motion: MotionLabel
    """
    The motion that was running when the collision happened.
    """

    trial: int
    """
    Number of the trial the collision happened in.
    """


@dataclass
class IllegalJointStateDatabase:
    """
    The illegal joint states recorded over a number of trials.
    """

    joint_names: list[str]
    """
    Names of the arm joints, in the order the positions of each state are given in.
    """

    illegal_joint_states: list[IllegalJointState] = field(default_factory=list)
    """
    The recorded states.
    """

    @classmethod
    def for_daisy_arms(cls) -> Self:
        """
        :return: An empty database over the joints of both DAiSy arms, left arm first,
            each ordered from shoulder to wrist.
        """
        return cls(
            joint_names=[
                joint.value
                for joint in (
                    DAiSyJoint.LEFT_SHOULDER_PAN,
                    DAiSyJoint.LEFT_SHOULDER_LIFT,
                    DAiSyJoint.LEFT_ELBOW,
                    DAiSyJoint.LEFT_WRIST_1,
                    DAiSyJoint.LEFT_WRIST_2,
                    DAiSyJoint.LEFT_WRIST_3,
                    DAiSyJoint.RIGHT_SHOULDER_PAN,
                    DAiSyJoint.RIGHT_SHOULDER_LIFT,
                    DAiSyJoint.RIGHT_ELBOW,
                    DAiSyJoint.RIGHT_WRIST_1,
                    DAiSyJoint.RIGHT_WRIST_2,
                    DAiSyJoint.RIGHT_WRIST_3,
                )
            ]
        )

    def joint_configurations(self, world: World) -> JointConfigurationSet:
        """
        :param world: The world holding the joints of the database.
        :return: The states as configurations of the world's joints, each compared on
            its colliding joints only.
        """
        return JointConfigurationSet(
            connections=[
                world.get_connection_by_name(name) for name in self.joint_names
            ],
            positions=np.array(
                [state.positions for state in self.illegal_joint_states], dtype=float
            ).reshape(-1, len(self.joint_names)),
            relevant_connections=np.array(
                [
                    [name in state.colliding_joint_names for name in self.joint_names]
                    for state in self.illegal_joint_states
                ],
                dtype=bool,
            ).reshape(-1, len(self.joint_names)),
        )

    def save(self, path: Path) -> None:
        """
        Write the database as JSON.

        :param path: The file to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self)))

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        :param path: A file written by :meth:`save`.
        :return: The database the file holds.
        """
        return from_json(json.loads(path.read_text()))

    @classmethod
    def load_merged(cls, paths: list[Path]) -> Self:
        """
        :param paths: Files written by :meth:`save`, over the same joints.
        :return: One database holding the states of all of them.
        """
        databases = [cls.load(path) for path in paths]
        return cls(
            joint_names=databases[0].joint_names,
            illegal_joint_states=[
                state
                for database in databases
                for state in database.illegal_joint_states
            ],
        )

    def extend_from_log(self, path: Path) -> None:
        """
        Add the states of a log written by :class:`ArmCollisionRecorder`.

        :param path: The log, one JSON encoded state per line.
        """
        self.illegal_joint_states.extend(
            from_json(json.loads(line)) for line in path.read_text().splitlines()
        )


# %% active motion


@dataclass
class ActiveMotionTracker(PlanCallback):
    """
    Keeps the label of the motion of a plan that started last.
    """

    stage: str = ""
    """
    The stage of the demo being performed, set by whoever performs it.
    """

    active_motion: MotionLabel | None = None
    """
    The motion that started last.
    """

    _motions_started: int = field(default=0, init=False)
    """
    Number of motions started in the current stage.
    """

    def begin_stage(self, stage: str) -> None:
        """
        Restart the numbering of motions for the next stage.

        :param stage: The name of the stage about to be performed.
        """
        # The plain value, so a label can be read without the module naming the stage.
        self.stage = str(stage)
        self._motions_started = 0

    def on_start(self, node: PlanNode) -> None:
        if not isinstance(node, MotionNode):
            return
        action = node.parent_action_node
        self.active_motion = MotionLabel(
            stage=self.stage,
            index=self._motions_started,
            action=type(action.action).__name__ if action is not None else "plan",
            motion=type(node.motion).__name__,
        )
        self._motions_started += 1


# %% arm bodies


@dataclass
class ArmCollisionPairs:
    """
    The bodies each arm of DAiSy moves, and the pairs of bodies involving them that the
    robot's collision rules check.
    """

    world: World
    """
    The world DAiSy is in.
    """

    robot: DAiSy
    """
    The robot whose arms are considered.
    """

    bodies_of_arm: dict[Arm, set[Body]] = field(init=False)
    """
    The bodies with collision geometry each arm's first joint moves, which includes the
    arm's gripper.
    """

    robot_bodies: set[Body] = field(init=False)
    """
    The bodies with collision geometry that belonged to the robot when this was built.
    """

    def __post_init__(self):
        self.bodies_of_arm = {
            arm: self._bodies_moved_by(arm)
            for arm in (self.robot.left_arm, self.robot.right_arm)
        }
        self.robot_bodies = set(self.robot.bodies_with_collision)

    def self_collision_matrix(self, distance: float) -> CollisionMatrix:
        """
        :param distance: The distance below which a pair counts as colliding.
        :return: The checked pairs of robot bodies of which at least one belongs to an
            arm.
        """
        arm_bodies = set().union(*self.bodies_of_arm.values())
        return CollisionMatrix(
            {
                CollisionCheck(check.body_a, check.body_b, distance)
                for check in self._checked_pairs()
                if {check.body_a, check.body_b} <= self.robot_bodies
                and {check.body_a, check.body_b} & arm_bodies
            }
        )

    def collision_matrix_of(self, arm: Arm, distance: float) -> CollisionMatrix:
        """
        :param arm: The arm whose bodies are checked.
        :param distance: The distance below which a pair counts as colliding.
        :return: The checked pairs, with the robot itself and with its surroundings, of
            which at least one body belongs to the arm.
        """
        return CollisionMatrix(
            {
                CollisionCheck(check.body_a, check.body_b, distance)
                for check in self._checked_pairs()
                if {check.body_a, check.body_b} & self.bodies_of_arm[arm]
            }
        )

    def _checked_pairs(self) -> set[CollisionCheck]:
        """
        :return: The pairs the collision rules of the world check now.
        """
        collision_manager = self.world.collision_manager
        collision_manager.update_collision_matrix(buffer=0.0)
        return collision_manager.collision_matrix.collision_checks

    def _bodies_moved_by(self, arm: Arm) -> set[Body]:
        """
        :return: The bodies with collision geometry that the arm's first joint moves.
        """
        first_joint = arm.active_connections[0]
        return {
            entity
            for entity in self.world.get_kinematic_structure_entities_of_branch(
                first_joint.child
            )
            if isinstance(entity, Body) and entity.has_collision()
        }


# %% recording


@dataclass(eq=False)
class ArmCollisionRecorder(StateChangeCallback):
    """
    Checks after every change of the world state whether a body of an arm touches
    another body of the robot, and appends the joint state of both arms to a log if it
    does.
    """

    robot: DAiSy = field(kw_only=True)
    """
    The robot whose arms are watched.
    """

    motion_tracker: ActiveMotionTracker = field(kw_only=True)
    """
    Names the motion a collision happened in.
    """

    log: Path = field(kw_only=True)
    """
    The file the states are appended to, one JSON encoded state per line.
    """

    trial: int = field(kw_only=True)
    """
    Number of the trial being recorded.
    """

    joint_names: list[str] = field(kw_only=True)
    """
    Names of the joints whose positions are logged, in the order they are logged in.
    """

    minimum_spacing: float = field(default=0.02, kw_only=True)
    """
    A colliding state closer than this to the previously logged one, in the largest
    joint difference in radians, is not logged again.
    """

    _arm_pairs: ArmCollisionPairs = field(init=False)
    """
    The bodies of the arms, and the pairs of them that are checked.
    """

    _collision_matrix: CollisionMatrix = field(init=False)
    """
    The body pairs of the robot involving an arm that may not touch.
    """

    _last_logged_positions: np.ndarray | None = field(init=False, default=None)
    """
    The positions of the state logged last.
    """

    def __post_init__(self):
        """
        Fixes which bodies belong to the robot, so objects attached to it later are not
        mistaken for parts of it.
        """
        self._arm_pairs = ArmCollisionPairs(world=self._world, robot=self.robot)
        self._collision_matrix = self._arm_pairs.self_collision_matrix(distance=0.0)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.touch()
        super().__post_init__()

    @property
    def connections(self) -> list[ActiveConnection1DOF]:
        """
        :return: The joints whose positions are logged, in the order they are logged in.
        """
        return [self._world.get_connection_by_name(name) for name in self.joint_names]

    def on_state_change(self, **kwargs) -> None:
        collision_detector = self._world.collision_manager.collision_detector
        contacts = [
            contact
            for contact in collision_detector.check_collisions(
                self._collision_matrix
            ).contacts
            if contact.distance <= 0.0
        ]
        if not contacts:
            return
        positions = np.array([connection.position for connection in self.connections])
        if (
            self._last_logged_positions is not None
            and np.max(np.abs(positions - self._last_logged_positions))
            < self.minimum_spacing
        ):
            return
        self._last_logged_positions = positions
        deepest = min(contacts, key=lambda contact: contact.distance)
        colliding_arms = self._arms_touched_by(deepest)
        state = IllegalJointState(
            positions=positions.tolist(),
            colliding_joint_names=[
                connection.name.name
                for arm in colliding_arms
                for connection in arm.active_connections
            ],
            collision_kind=self._collision_kind(deepest, colliding_arms),
            first_body=deepest.body_a.name.name,
            second_body=deepest.body_b.name.name,
            penetration_depth=-float(deepest.distance),
            motion=self.motion_tracker.active_motion,
            trial=self.trial,
        )
        with self.log.open("a") as log:
            log.write(json.dumps(to_json(state)) + "\n")

    def _arms_touched_by(self, contact: ClosestPoints) -> list[Arm]:
        """
        :return: The arms a body of the contact belongs to.
        """
        bodies = {contact.body_a, contact.body_b}
        return [
            arm
            for arm, arm_bodies in self._arm_pairs.bodies_of_arm.items()
            if bodies & arm_bodies
        ]

    def _collision_kind(
        self, contact: ClosestPoints, colliding_arms: list[Arm]
    ) -> CollisionKind:
        """
        :return: Which parts of the robot the contact is between.
        """
        if len(colliding_arms) == 2:
            return CollisionKind.BETWEEN_ARMS
        arm_bodies = self._arm_pairs.bodies_of_arm[colliding_arms[0]]
        if {contact.body_a, contact.body_b} <= arm_bodies:
            return CollisionKind.WITHIN_ARM
        return CollisionKind.ARM_AND_ROBOT_BODY
