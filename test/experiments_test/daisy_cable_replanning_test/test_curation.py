"""
Choosing recorded illegal joint states to keep in a permanent database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.daisy_cable_replanning.collisions import (
    CollisionKind,
    IllegalJointState,
    IllegalJointStateDatabase,
    MotionLabel,
)
from experiments.daisy_cable_replanning.curation import IllegalJointStateCuration
from experiments.daisy_cable_replanning.exceptions import IncompatibleJointNames

# %% helpers


def _state(first: float, second: float, trial: int = 0) -> IllegalJointState:
    """
    :return: An illegal joint state of both DAiSy arms, with the first two joints at the
        positions and every other joint at zero.
    """
    joint_names = IllegalJointStateDatabase.for_daisy_arms().joint_names
    return IllegalJointState(
        positions=[first, second] + [0.0] * (len(joint_names) - 2),
        colliding_joint_names=joint_names,
        collision_kind=CollisionKind.WITHIN_ARM,
        first_body="forearm",
        second_body="gripper",
        penetration_depth=0.01,
        motion=MotionLabel(stage="rehang", index=1, action="Rehang", motion="Move"),
        trial=trial,
    )


def _database(*states: IllegalJointState) -> IllegalJointStateDatabase:
    """
    :return: A database of both DAiSy arms holding the states.
    """
    database = IllegalJointStateDatabase.for_daisy_arms()
    database.illegal_joint_states.extend(states)
    return database


# %% choosing


def test_recorded_states_already_kept_start_out_chosen():
    recorded = _database(_state(0.1, 0.2), _state(0.3, 0.4), _state(0.5, 0.6))
    kept = _database(_state(0.3, 0.4, trial=7))

    curation = IllegalJointStateCuration.of(recorded, kept)

    assert curation.chosen == [False, True, False]


def test_curated_database_keeps_states_the_recording_does_not_hold():
    only_kept = _state(0.9, 0.9)
    recorded = _database(_state(0.1, 0.2), _state(0.3, 0.4))
    curation = IllegalJointStateCuration.of(recorded, _database(only_kept))
    curation.chosen[0] = True

    curated = curation.curated()

    assert curated.illegal_joint_states == [
        only_kept,
        recorded.illegal_joint_states[0],
    ]


def test_unchoosing_a_kept_recorded_state_removes_it():
    recorded = _database(_state(0.1, 0.2), _state(0.3, 0.4))
    curation = IllegalJointStateCuration.of(recorded, _database(_state(0.3, 0.4)))
    curation.chosen[1] = False

    assert curation.curated().illegal_joint_states == []


def test_a_state_chosen_and_kept_already_is_kept_once():
    recorded = _database(_state(0.3, 0.4))
    curation = IllegalJointStateCuration.of(recorded, _database(_state(0.3, 0.4)))

    assert curation.curated().illegal_joint_states == recorded.illegal_joint_states


def test_databases_of_different_joints_cannot_be_curated():
    kept = IllegalJointStateDatabase(joint_names=["other_joint"])

    with pytest.raises(IncompatibleJointNames):
        IllegalJointStateCuration.of(_database(_state(0.1, 0.2)), kept)


# %% files


def test_a_permanent_file_that_does_not_exist_yet_is_empty(tmp_path: Path):
    recorded = _database(_state(0.1, 0.2))
    recorded.save(tmp_path / "recorded.json")

    curation = IllegalJointStateCuration.load(
        tmp_path / "recorded.json", tmp_path / "permanent.json"
    )

    assert curation.chosen == [False]
    assert curation.kept.joint_names == recorded.joint_names


def test_saved_curation_is_chosen_when_loaded_again(tmp_path: Path):
    _database(_state(0.1, 0.2), _state(0.3, 0.4)).save(tmp_path / "recorded.json")
    curation = IllegalJointStateCuration.load(
        tmp_path / "recorded.json", tmp_path / "permanent.json"
    )
    curation.chosen[1] = True
    curation.curated().save(tmp_path / "permanent.json")

    reloaded = IllegalJointStateCuration.load(
        tmp_path / "recorded.json", tmp_path / "permanent.json"
    )

    assert reloaded.chosen == [False, True]
