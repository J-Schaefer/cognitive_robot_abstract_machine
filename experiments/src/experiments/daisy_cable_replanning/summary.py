from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from experiments.daisy_cable_replanning.collisions import IllegalJointStateDatabase
from experiments.daisy_cable_replanning.trials import CableDemoExperiment, TrialResult
from krrood.adapters.json_serializer import from_json
from typing_extensions import Self

# %% one experiment


@dataclass
class ExperimentOutcome:
    """
    What a finished run of :class:`CableDemoExperiment` wrote: the result of each trial
    and the illegal joint states the trials ran into.
    """

    results: dict[int, TrialResult]
    """
    The result of every trial, by seed.
    """

    database: IllegalJointStateDatabase
    """
    The illegal joint states all trials ran into.
    """

    @classmethod
    def load(cls, output_directory: Path) -> Self:
        """
        :param output_directory: The output directory of the experiment.
        :return: What the experiment wrote there.
        """
        experiment = CableDemoExperiment(output_directory=output_directory)
        results = from_json(json.loads(experiment.results_path.read_text()))
        return cls(
            results={result.seed: result for result in results},
            database=IllegalJointStateDatabase.load(experiment.database_path),
        )

    @property
    def colliding_seeds(self) -> set[int]:
        """
        :return: The seeds of the trials that ran into an illegal joint state.
        """
        return {
            seed
            for seed, result in self.results.items()
            if result.illegal_joint_states > 0
        }

    @property
    def completed_seeds(self) -> set[int]:
        """
        :return: The seeds of the trials that performed every stage.
        """
        return {seed for seed, result in self.results.items() if result.failure is None}

    @property
    def successful_seeds(self) -> set[int]:
        """
        :return: The seeds of the trials that performed every stage without running
            into an illegal joint state.
        """
        return self.completed_seeds - self.colliding_seeds

    @property
    def replanned_seeds(self) -> set[int]:
        """
        :return: The seeds of the trials that replanned a motion.
        """
        return {
            seed
            for seed, result in self.results.items()
            if result.replanned_motions > 0
        }

    def describe(self) -> str:
        """
        :return: A text summary of the trials, their failures and the collisions.
        """
        states = self.database.illegal_joint_states
        lines = [
            (
                f"trials: {len(self.results)}, completed: {len(self.completed_seeds)}, "
                f"colliding: {len(self.colliding_seeds)}, "
                f"replanning: {len(self.replanned_seeds)} trials / "
                f"{sum(result.replanned_motions for result in self.results.values())} "
                "motions"
            ),
            f"illegal joint states: {len(states)}",
            "collisions by kind:",
            *self._counted(Counter(state.collision_kind.value for state in states)),
            "trials colliding per motion:",
            *self._counted(
                Counter(
                    str(motion)
                    for _, motion in {(state.trial, state.motion) for state in states}
                )
            ),
            "colliding body pairs:",
            *self._counted(
                Counter(
                    " / ".join(sorted((state.first_body, state.second_body)))
                    for state in states
                )
            ),
            "failures (message up to its first joint values):",
            *self._counted(
                Counter(
                    result.failure.split("[")[0]
                    for result in self.results.values()
                    if result.failure is not None
                )
            ),
        ]
        return "\n".join(lines)

    @staticmethod
    def _counted(counter: Counter) -> list[str]:
        """
        :return: One indented line per counted entry, the most frequent first.
        """
        return [f"  {count:5d}  {entry}" for entry, count in counter.most_common()]


# %% comparison


@dataclass
class ReplanningComparison:
    """
    The same seeds performed once as the demo is and once with replanning.
    """

    baseline: ExperimentOutcome
    """
    The trials performed as the demo is.
    """

    replanning: ExperimentOutcome
    """
    The trials performed with replanning.
    """

    def describe(self) -> str:
        """
        :return: A text summary of how replanning changed the trials of the seeds both
            experiments ran.
        """
        seeds = self.baseline.results.keys() & self.replanning.results.keys()
        colliding = self.baseline.colliding_seeds & seeds
        clear = (self.baseline.completed_seeds - self.baseline.colliding_seeds) & seeds
        return "\n".join(
            [
                f"seeds in both: {len(seeds)}",
                f"colliding without replanning: {len(colliding)}",
                (
                    "  of those with replanning: "
                    f"replanned {len(colliding & self.replanning.replanned_seeds)}, "
                    f"collision-free {len(colliding - self.replanning.colliding_seeds)}, "
                    f"completed {len(colliding & self.replanning.completed_seeds)}"
                ),
                f"completed collision-free without replanning: {len(clear)}",
                (
                    "  of those with replanning: "
                    f"replanned {len(clear & self.replanning.replanned_seeds)}, "
                    f"completed {len(clear & self.replanning.completed_seeds)}"
                ),
                (
                    "completed: "
                    f"{len(self.baseline.completed_seeds & seeds)} without, "
                    f"{len(self.replanning.completed_seeds & seeds)} with replanning"
                ),
                (
                    "completed without any collision: "
                    f"{len(self.baseline.successful_seeds & seeds)} without, "
                    f"{len(self.replanning.successful_seeds & seeds)} with replanning"
                ),
                (
                    "illegal joint states: "
                    f"{len(self.baseline.database.illegal_joint_states)} without, "
                    f"{len(self.replanning.database.illegal_joint_states)} with replanning"
                ),
            ]
        )
