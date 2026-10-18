"""
Leaving illegal joint states that repeat an earlier pose out of a database.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.daisy_cable_replanning.collisions import IllegalJointStateDatabase
from experiments.daisy_cable_replanning.trials import (
    CableDemoExperiment,
    CableDemoTrial,
    TrialFiles,
)
from krrood.adapters.json_serializer import to_json

from .synthetic_states import database_of, state_at

# %% database


def test_a_state_within_the_tolerance_of_an_earlier_one_is_left_out():
    first = state_at(0.10, 0.20)
    database = database_of(first, state_at(0.11, 0.19), state_at(0.10, 0.205))

    assert database.without_duplicate_poses(0.02).illegal_joint_states == [first]


def test_a_state_beyond_the_tolerance_is_kept():
    database = database_of(state_at(0.10, 0.20), state_at(0.13, 0.20))

    distinct = database.without_duplicate_poses(0.02)

    assert distinct.illegal_joint_states == database.illegal_joint_states


def test_joints_that_do_not_collide_do_not_tell_poses_apart():
    joint_names = IllegalJointStateDatabase.for_daisy_arms().joint_names
    first = state_at(0.10, 0.20, colliding_joint_names=joint_names[:1])
    database = database_of(
        first, state_at(0.10, 1.50, colliding_joint_names=joint_names[:1])
    )

    assert database.without_duplicate_poses(0.02).illegal_joint_states == [first]


def test_states_of_different_colliding_joints_are_different_poses():
    joint_names = IllegalJointStateDatabase.for_daisy_arms().joint_names
    database = database_of(
        state_at(0.10, 0.20, colliding_joint_names=joint_names[:6]),
        state_at(0.10, 0.20, colliding_joint_names=joint_names[6:]),
    )

    distinct = database.without_duplicate_poses(0.02)

    assert distinct.illegal_joint_states == database.illegal_joint_states


# %% experiment


def _trial_with_log(tmp_path: Path) -> CableDemoTrial:
    """
    :return: A trial whose collision log holds the same pose twice and one other.
    """
    trial = CableDemoTrial(
        number=0, seed=0, files=TrialFiles(directory=tmp_path, number=0)
    )
    states = [state_at(0.10, 0.20), state_at(0.105, 0.20), state_at(0.50, 0.20)]
    trial.files.collisions.write_text(
        "".join(json.dumps(to_json(state)) + "\n" for state in states)
    )
    return trial


def test_experiment_leaves_repeated_poses_out_by_default(tmp_path: Path):
    trial = _trial_with_log(tmp_path)
    experiment = CableDemoExperiment(output_directory=tmp_path)

    database = experiment.collected_database([trial])

    assert experiment.removes_duplicate_poses
    assert len(database.illegal_joint_states) == 2


def test_experiment_keeps_every_state_when_told_to(tmp_path: Path):
    trial = _trial_with_log(tmp_path)
    experiment = CableDemoExperiment(
        output_directory=tmp_path, removes_duplicate_poses=False
    )

    database = experiment.collected_database([trial])

    assert len(database.illegal_joint_states) == 3
