"""
Choosing recorded illegal joint states to keep in a permanent database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.daisy_cable_replanning.collisions import IllegalJointStateDatabase
from experiments.daisy_cable_replanning.curation import IllegalJointStateCuration
from experiments.daisy_cable_replanning.exceptions import IncompatibleJointNames

from .synthetic_states import database_of, state_at

# %% choosing


def test_recorded_states_already_kept_start_out_chosen():
    recorded = database_of(state_at(0.1, 0.2), state_at(0.3, 0.4), state_at(0.5, 0.6))
    kept = database_of(state_at(0.3, 0.4, trial=7))

    curation = IllegalJointStateCuration.of(recorded, kept)

    assert curation.chosen == [False, True, False]


def test_curated_database_keeps_states_the_recording_does_not_hold():
    only_kept = state_at(0.9, 0.9)
    recorded = database_of(state_at(0.1, 0.2), state_at(0.3, 0.4))
    curation = IllegalJointStateCuration.of(recorded, database_of(only_kept))
    curation.chosen[0] = True

    curated = curation.curated()

    assert curated.illegal_joint_states == [
        only_kept,
        recorded.illegal_joint_states[0],
    ]


def test_unchoosing_a_kept_recorded_state_removes_it():
    recorded = database_of(state_at(0.1, 0.2), state_at(0.3, 0.4))
    curation = IllegalJointStateCuration.of(recorded, database_of(state_at(0.3, 0.4)))
    curation.chosen[1] = False

    assert curation.curated().illegal_joint_states == []


def test_a_state_chosen_and_kept_already_is_kept_once():
    recorded = database_of(state_at(0.3, 0.4))
    curation = IllegalJointStateCuration.of(recorded, database_of(state_at(0.3, 0.4)))

    assert curation.curated().illegal_joint_states == recorded.illegal_joint_states


def test_databases_of_different_joints_cannot_be_curated():
    kept = IllegalJointStateDatabase(joint_names=["other_joint"])

    with pytest.raises(IncompatibleJointNames):
        IllegalJointStateCuration.of(database_of(state_at(0.1, 0.2)), kept)


# %% files


def test_a_permanent_file_that_does_not_exist_yet_is_empty(tmp_path: Path):
    recorded = database_of(state_at(0.1, 0.2))
    recorded.save(tmp_path / "recorded.json")

    curation = IllegalJointStateCuration.load(
        tmp_path / "recorded.json", tmp_path / "permanent.json"
    )

    assert curation.chosen == [False]
    assert curation.kept.joint_names == recorded.joint_names


def test_saved_curation_is_chosen_when_loaded_again(tmp_path: Path):
    database_of(state_at(0.1, 0.2), state_at(0.3, 0.4)).save(tmp_path / "recorded.json")
    curation = IllegalJointStateCuration.load(
        tmp_path / "recorded.json", tmp_path / "permanent.json"
    )
    curation.chosen[1] = True
    curation.curated().save(tmp_path / "permanent.json")

    reloaded = IllegalJointStateCuration.load(
        tmp_path / "recorded.json", tmp_path / "permanent.json"
    )

    assert reloaded.chosen == [False, True]
