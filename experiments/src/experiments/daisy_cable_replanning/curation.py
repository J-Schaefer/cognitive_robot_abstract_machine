from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from experiments.daisy_cable_replanning.collisions import (
    IllegalJointState,
    IllegalJointStateDatabase,
)
from experiments.daisy_cable_replanning.exceptions import IncompatibleJointNames
from typing_extensions import Self

# %% kept databases


class KeptDatabase(Enum):
    """
    Databases of illegal joint states kept for good, by the robot setup they belong to.
    """

    DAISY_CABLE_WORKCELL = (
        Path(__file__).parent
        / "resources"
        / "daisy_cable_workcell_illegal_joint_states.json"
    )
    """
    The illegal joint states of DAiSy with the cable post and hanger of the cable demo.
    """


# %% curation


@dataclass
class IllegalJointStateCuration:
    """
    A choice of which recorded illegal joint states to keep in a database that is kept
    for good.

    A recorded state that the kept database already holds starts out chosen. The
    curated database holds the kept states the recording does not contain, followed by
    the chosen recorded states, so unchoosing a recorded state removes it from the kept
    database.
    """

    recorded: IllegalJointStateDatabase
    """
    The illegal joint states of a recording.
    """

    kept: IllegalJointStateDatabase
    """
    The illegal joint states kept so far.
    """

    chosen: list[bool]
    """
    Per recorded state, whether it is kept.
    """

    _kept_joint_states: set[tuple[float, ...]] = field(init=False, repr=False)
    """
    The rounded positions of the kept states.
    """

    def __post_init__(self):
        self._kept_joint_states = {
            state.rounded_positions for state in self.kept.illegal_joint_states
        }

    @classmethod
    def of(
        cls, recorded: IllegalJointStateDatabase, kept: IllegalJointStateDatabase
    ) -> Self:
        """
        :param recorded: The illegal joint states of a recording.
        :param kept: The illegal joint states kept so far.
        :return: The curation in which the recorded states the kept database already
            holds are chosen.
        :raises IncompatibleJointNames: If the databases give positions of different
            joints.
        """
        if recorded.joint_names != kept.joint_names:
            raise IncompatibleJointNames(
                recorded_joint_names=recorded.joint_names,
                kept_joint_names=kept.joint_names,
            )
        curation = cls(recorded=recorded, kept=kept, chosen=[])
        curation.chosen = [
            curation.is_kept_already(state) for state in recorded.illegal_joint_states
        ]
        return curation

    @classmethod
    def load(cls, recorded_path: Path, kept_path: Path) -> Self:
        """
        :param recorded_path: A database written by a recording.
        :param kept_path: The database kept so far; a file that does not exist yet
            counts as an empty database.
        :return: The curation of the two databases.
        """
        recorded = IllegalJointStateDatabase.load(recorded_path)
        kept = (
            IllegalJointStateDatabase.load(kept_path)
            if kept_path.exists()
            else IllegalJointStateDatabase(joint_names=recorded.joint_names)
        )
        return cls.of(recorded, kept)

    def is_kept_already(self, state: IllegalJointState) -> bool:
        """
        :return: Whether the kept database already holds the joint state.
        """
        return state.rounded_positions in self._kept_joint_states

    def curated(self) -> IllegalJointStateDatabase:
        """
        :return: The kept states the recording does not contain, followed by the
            chosen recorded states.
        """
        recorded_joint_states = {
            state.rounded_positions for state in self.recorded.illegal_joint_states
        }
        return IllegalJointStateDatabase(
            joint_names=self.kept.joint_names,
            illegal_joint_states=[
                state
                for state in self.kept.illegal_joint_states
                if state.rounded_positions not in recorded_joint_states
            ]
            + [
                state
                for state, chosen in zip(
                    self.recorded.illegal_joint_states, self.chosen
                )
                if chosen
            ],
        )
