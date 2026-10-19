"""
Counting the illegal joint states of runs: logged, repeated, written and within the
guard's reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments.daisy_cable_replanning.collisions import (
    IllegalJointState,
    IllegalJointStateDatabase,
)
from experiments.daisy_cable_replanning.joint_state_metrics import (
    PoseNeighbourhood,
    RunJointStateCounts,
)
from experiments.daisy_cable_replanning.trials import CableDemoExperiment, TrialFiles
from krrood.adapters.json_serializer import to_json

from .synthetic_states import database_of, state_at

# %% distances


def test_distance_counts_only_the_colliding_joints_of_the_database_state():
    joint_names = IllegalJointStateDatabase.for_daisy_arms().joint_names
    neighbourhood = PoseNeighbourhood.of(
        database_of(state_at(0.0, 0.0, colliding_joint_names=joint_names[:1]))
    )
    queried = np.zeros((1, len(joint_names)))
    queried[0, :2] = [0.1, 5.0]

    assert neighbourhood.nearest_distances(queried) == pytest.approx([0.1])


def test_distance_to_the_nearest_of_several_states():
    neighbourhood = PoseNeighbourhood.of(
        database_of(state_at(0.0, 0.0), state_at(1.0, 0.0))
    )
    queried = PoseNeighbourhood.of(database_of(state_at(0.9, 0.0))).positions

    assert neighbourhood.nearest_distances(queried) == pytest.approx([0.1])


def test_nothing_is_near_an_empty_database():
    queried = PoseNeighbourhood.of(database_of(state_at(0.0, 0.0))).positions

    assert np.isinf(
        PoseNeighbourhood.of(database_of()).nearest_distances(queried)
    ).all()


# %% one run


def _run_directory(tmp_path: Path, logged: list[IllegalJointState]) -> Path:
    """
    :return: The output directory of a run whose single trial logged the states and
        whose database holds the first of them.
    """
    files = TrialFiles(directory=tmp_path / "trials", number=0)
    files.directory.mkdir(parents=True)
    files.collisions.write_text(
        "".join(json.dumps(to_json(state)) + "\n" for state in logged)
    )
    database_of(logged[0]).save(
        CableDemoExperiment(output_directory=tmp_path).database_path
    )
    return tmp_path


def test_run_counts_logged_distinct_and_written_states(tmp_path: Path):
    logged = [state_at(0.10, 0.20), state_at(0.11, 0.20), state_at(0.50, 0.20)]

    counts = RunJointStateCounts.of(
        "run",
        _run_directory(tmp_path, logged),
        [],
        tolerance=0.02,
        trigger_distance=0.15,
    )

    assert (counts.logged, counts.distinct, counts.written) == (3, 2, 1)
    assert counts.duplicate_share == pytest.approx(1 / 3)


def test_run_counts_the_states_within_the_guard_reach_of_the_databases_used(
    tmp_path: Path,
):
    logged = [state_at(0.10, 0.20), state_at(0.50, 0.20)]
    used = database_of(state_at(0.0, 0.20))

    counts = RunJointStateCounts.of(
        "run",
        _run_directory(tmp_path, logged),
        [used],
        tolerance=0.02,
        trigger_distance=0.15,
    )

    assert counts.within_guard_reach == 1
    assert counts.databases_used == 1
