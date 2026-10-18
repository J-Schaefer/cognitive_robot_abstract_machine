"""
Illegal joint states made up for tests, of both DAiSy arms.
"""

from __future__ import annotations

from experiments.daisy_cable_replanning.collisions import (
    CollisionKind,
    IllegalJointState,
    IllegalJointStateDatabase,
    MotionLabel,
)

# %% synthetic illegal joint states


def state_at(
    first: float,
    second: float,
    trial: int = 0,
    colliding_joint_names: list[str] | None = None,
) -> IllegalJointState:
    """
    :return: An illegal joint state of both DAiSy arms, with the first two joints at the
        positions and every other joint at zero, in which the given joints, or all of
        them, collide.
    """
    joint_names = IllegalJointStateDatabase.for_daisy_arms().joint_names
    return IllegalJointState(
        positions=[first, second] + [0.0] * (len(joint_names) - 2),
        colliding_joint_names=(
            joint_names if colliding_joint_names is None else colliding_joint_names
        ),
        collision_kind=CollisionKind.WITHIN_ARM,
        first_body="forearm",
        second_body="gripper",
        penetration_depth=0.01,
        motion=MotionLabel(stage="rehang", index=1, action="Rehang", motion="Move"),
        trial=trial,
    )


def database_of(*states: IllegalJointState) -> IllegalJointStateDatabase:
    """
    :return: A database of both DAiSy arms holding the states.
    """
    database = IllegalJointStateDatabase.for_daisy_arms()
    database.illegal_joint_states.extend(states)
    return database
