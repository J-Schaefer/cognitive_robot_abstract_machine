"""
Exceptions of the DAiSy cable replanning experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class NoJointPathFound(DataclassException):
    """
    Raised when the sampling-based planner finds no path to the goal that is free of
    collisions and clear of the illegal joint states.
    """

    start: list[float]
    """
    Positions of the planned joints at the start.
    """

    goals: list[list[float]]
    """
    The alternative positions of the planned joints at the goal.
    """

    planning_time: float
    """
    How long the planner searched, in seconds.
    """

    def error_message(self) -> str:
        return (
            f"No collision-free joint path from {self.start} to any of "
            f"{len(self.goals)} goal configurations was found within "
            f"{self.planning_time} s."
        )

    def suggest_correction(self) -> str:
        return (
            "Give the planner more time or more goal seeds, lower the clearance to the "
            "illegal joint states, or check whether start and goals are themselves in "
            "collision."
        )


@dataclass
class IncompatibleJointNames(DataclassException):
    """
    Raised when two databases of illegal joint states give positions of different
    joints, so their states cannot be compared.
    """

    recorded_joint_names: list[str]
    """
    The joints of the recorded database.
    """

    kept_joint_names: list[str]
    """
    The joints of the kept database.
    """

    def error_message(self) -> str:
        return (
            f"The recorded database gives positions of {self.recorded_joint_names}, "
            f"the kept one of {self.kept_joint_names}."
        )

    def suggest_correction(self) -> str:
        return "Choose a kept database recorded for the same robot."
