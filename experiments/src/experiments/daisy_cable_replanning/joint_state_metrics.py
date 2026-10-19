from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from experiments.daisy_cable_replanning.collisions import IllegalJointStateDatabase
from experiments.daisy_cable_replanning.trials import CableDemoExperiment
from typing_extensions import Self

# %% distances between databases


@dataclass
class PoseNeighbourhood:
    """
    The illegal joint states of a database, as arrays, compared with other joint states
    on the joints each of its states marks as colliding.
    """

    positions: np.ndarray
    """
    One state per row, one joint per column.
    """

    colliding: np.ndarray
    """
    Per state and joint, whether the joint collides in the state.
    """

    @classmethod
    def of(cls, database: IllegalJointStateDatabase) -> Self:
        """
        :return: The states of the database as arrays.
        """
        joint_names = database.joint_names
        return cls(
            positions=np.array(
                [state.positions for state in database.illegal_joint_states],
                dtype=float,
            ).reshape(-1, len(joint_names)),
            colliding=np.array(
                [
                    [name in state.colliding_joint_names for name in joint_names]
                    for state in database.illegal_joint_states
                ],
                dtype=bool,
            ).reshape(-1, len(joint_names)),
        )

    def nearest_distances(
        self, positions: np.ndarray, batch_size: int = 32
    ) -> np.ndarray:
        """
        :param positions: Joint states, one per row.
        :param batch_size: How many joint states are compared at once, bounding the
            memory of the comparison.
        :return: Per joint state, the distance the guard measures to the nearest state
            of this neighbourhood; infinite if it is empty.
        """
        if len(self.positions) == 0:
            return np.full(len(positions), np.inf)
        distances = []
        for start in range(0, len(positions), batch_size):
            batch = positions[start : start + batch_size]
            differences = np.where(
                self.colliding[None],
                np.abs(self.positions[None] - batch[:, None]),
                0.0,
            )
            distances.append(differences.max(axis=2).min(axis=1))
        return np.concatenate(distances)

    def guard_seconds_per_check(self, repetitions: int = 200) -> float:
        """
        :return: How long the guard takes to measure the distance of one joint state to
            this neighbourhood, which it does once per control cycle, in seconds.
        """
        joint_state = self.positions[0] if len(self.positions) else np.zeros(12)
        start = time.perf_counter()
        for _ in range(repetitions):
            np.min(
                np.max(
                    np.where(self.colliding, np.abs(self.positions - joint_state), 0.0),
                    axis=1,
                    initial=0.0,
                ),
                initial=np.inf,
            )
        return (time.perf_counter() - start) / repetitions


# %% one run


@dataclass
class RunJointStateCounts:
    """
    How many illegal joint states a run of :class:`CableDemoExperiment` logged and
    wrote, and how many of its poses were new to the databases it replanned around.
    """

    name: str
    """
    Name of the run within its series.
    """

    logged: int
    """
    The states its trials logged.
    """

    written: int
    """
    The states of the database it wrote.
    """

    distinct: int
    """
    The distinct poses among the logged states, at the duplicate tolerance.
    """

    within_guard_reach: int
    """
    The logged states within the guard's trigger distance of a state of the databases
    the run replanned around: collisions the guard knew of and still did not prevent.
    """

    databases_used: int
    """
    The states of the databases the run replanned around.
    """

    @property
    def duplicate_share(self) -> float:
        """
        :return: The share of the logged states that repeat an earlier pose.
        """
        return 1.0 - self.distinct / self.logged if self.logged else 0.0

    @classmethod
    def of(
        cls,
        name: str,
        output_directory: Path,
        databases_used: list[IllegalJointStateDatabase],
        tolerance: float,
        trigger_distance: float,
    ) -> Self:
        """
        :param name: Name of the run within its series.
        :param output_directory: The output directory of the run.
        :param databases_used: The databases the run replanned around.
        :param tolerance: The largest joint difference of two states of one pose.
        :param trigger_distance: The distance at which the guard stops a motion.
        :return: The counts of the run.
        """
        logged = IllegalJointStateDatabase.for_daisy_arms()
        for log in sorted((output_directory / "trials").glob("*.collisions.jsonl")):
            logged.extend_from_log(log)
        distinct = logged.without_duplicate_poses(tolerance)
        used = IllegalJointStateDatabase.for_daisy_arms()
        for database in databases_used:
            used.illegal_joint_states.extend(database.illegal_joint_states)
        distances = PoseNeighbourhood.of(used).nearest_distances(
            PoseNeighbourhood.of(logged).positions
        )
        written = IllegalJointStateDatabase.load(
            CableDemoExperiment(output_directory=output_directory).database_path
        )
        return cls(
            name=name,
            logged=len(logged.illegal_joint_states),
            written=len(written.illegal_joint_states),
            distinct=len(distinct.illegal_joint_states),
            within_guard_reach=int(np.sum(distances < trigger_distance)),
            databases_used=len(used.illegal_joint_states),
        )


# %% one series


@dataclass
class SeriesJointStateMetrics:
    """
    The joint state counts of the runs of one series as ``run_series.sh`` writes it, the
    coverage of the held-out collisions by the training databases, and what checking
    them costs the guard.
    """

    runs: list[RunJointStateCounts]
    """
    The counts of every run, in the order the series ran them.
    """

    training_database_states: int
    """
    The states of all databases the held-out replanning used.
    """

    held_out_coverage: float
    """
    The share of the illegal joint states of the held-out baseline within the guard's
    trigger distance of a state of those databases.
    """

    guard_seconds_per_check: float
    """
    How long the guard takes per control cycle with those databases, in seconds.
    """

    @classmethod
    def of(
        cls,
        series_directory: Path,
        rounds: int,
        extra_databases: list[Path],
        tolerance: float,
        trigger_distance: float,
    ) -> Self:
        """
        :param series_directory: The output directory of the series.
        :param rounds: How many replanning rounds the series ran.
        :param extra_databases: The databases of earlier series every replanning used.
        :param tolerance: The largest joint difference of two states of one pose.
        :param trigger_distance: The distance at which the guard stops a motion.
        :return: The metrics of the series.
        """
        extra = [IllegalJointStateDatabase.load(path) for path in extra_databases]

        def database_of(run: str) -> IllegalJointStateDatabase:
            return IllegalJointStateDatabase.load(
                CableDemoExperiment(
                    output_directory=series_directory / run
                ).database_path
            )

        training = [*extra, database_of("record")]
        runs = [
            RunJointStateCounts.of(
                "record", series_directory / "record", [], tolerance, trigger_distance
            ),
            RunJointStateCounts.of(
                "heldout_record",
                series_directory / "heldout_record",
                [],
                tolerance,
                trigger_distance,
            ),
        ]
        for round_number in range(1, rounds + 1):
            name = f"replan_round_{round_number}"
            runs.append(
                RunJointStateCounts.of(
                    name,
                    series_directory / name,
                    list(training),
                    tolerance,
                    trigger_distance,
                )
            )
            training.append(database_of(name))
        runs.append(
            RunJointStateCounts.of(
                "heldout_replan",
                series_directory / "heldout_replan",
                training,
                tolerance,
                trigger_distance,
            )
        )
        all_training = IllegalJointStateDatabase.for_daisy_arms()
        for database in training:
            all_training.illegal_joint_states.extend(database.illegal_joint_states)
        neighbourhood = PoseNeighbourhood.of(all_training)
        held_out = IllegalJointStateDatabase.for_daisy_arms()
        for log in sorted(
            (series_directory / "heldout_record" / "trials").glob("*.collisions.jsonl")
        ):
            held_out.extend_from_log(log)
        distances = neighbourhood.nearest_distances(
            PoseNeighbourhood.of(held_out).positions
        )
        return cls(
            runs=runs,
            training_database_states=len(all_training.illegal_joint_states),
            held_out_coverage=float(np.mean(distances < trigger_distance)),
            guard_seconds_per_check=neighbourhood.guard_seconds_per_check(),
        )

    def describe(self) -> str:
        """
        :return: The metrics as a Markdown table and a few lines below it.
        """
        lines = [
            (
                "| run | states replanned around | logged | duplicate share | "
                "distinct poses | written | logged within guard reach |"
            ),
            "|---|---|---|---|---|---|---|",
            *(
                f"| {run.name} | {run.databases_used:,} | {run.logged:,} | "
                f"{run.duplicate_share:.0%} | {run.distinct:,} | {run.written:,} | "
                f"{run.within_guard_reach:,} |"
                for run in self.runs
            ),
            "",
            f"training databases: {self.training_database_states:,} states",
            (
                f"held-out baseline collisions within the trigger distance of them: "
                f"{self.held_out_coverage:.1%}"
            ),
            f"guard check per control cycle: {self.guard_seconds_per_check * 1e3:.2f} ms",
        ]
        return "\n".join(lines)
