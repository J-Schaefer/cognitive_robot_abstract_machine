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

    goal: list[float]
    """
    Positions of the planned joints at the goal.
    """

    planning_time: float
    """
    How long the planner searched, in seconds.
    """

    def error_message(self) -> str:
        return (
            f"No collision-free joint path from {self.start} to {self.goal} was found "
            f"within {self.planning_time} s."
        )

    def suggest_correction(self) -> str:
        return (
            "Give the planner more time, lower the clearance to the illegal joint "
            "states, or check whether start and goal are themselves in collision."
        )
