"""
Exceptions of the DAiSy cable replanning experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from semantic_digital_twin.collision_checking.collision_detector import (
        ClosestPoints,
    )


@dataclass
class RejectedGoalConfiguration:
    """
    A goal configuration of the planned joints and why the planner may not end in it.
    """

    positions: list[float]
    """
    Positions of the planned joints at the goal.
    """

    distance_to_illegal_configurations: float
    """
    Joint distance, in radians, to the nearest illegal joint configuration.
    """

    contacts: list[ClosestPoints]
    """
    The pairs of bodies closer than their allowed distance at the goal.
    """

    def reason(self, clearance: float) -> str:
        """
        :param clearance: The joint distance the goal has to keep from every illegal
            joint configuration.
        :return: Why the goal was rejected.
        """
        reasons = [
            f"{contact.body_a.name.name} - {contact.body_b.name.name} at "
            f"{contact.distance * 1000:.1f} mm"
            for contact in self.contacts
        ]
        if self.distance_to_illegal_configurations < clearance:
            reasons.append(
                f"{self.distance_to_illegal_configurations:.3f} rad from an illegal "
                f"joint configuration"
            )
        return ", ".join(reasons)


@dataclass
class NoValidGoalConfiguration(DataclassException):
    """
    Raised when every goal configuration given to the planner is in collision or too
    close to an illegal joint configuration, so no path is searched for.
    """

    start: list[float]
    """
    Positions of the planned joints at the start.
    """

    rejected_goals: list[RejectedGoalConfiguration]
    """
    Every goal configuration and why it was rejected.
    """

    clearance: float
    """
    The joint distance, in radians, a goal has to keep from every illegal joint
    configuration.
    """

    def error_message(self) -> str:
        reasons = "\n".join(
            f"  goal {index}: {goal.reason(self.clearance)}"
            for index, goal in enumerate(self.rejected_goals)
        )
        return (
            f"None of the {len(self.rejected_goals)} goal configurations for the path "
            f"from {self.start} is valid, so no path was searched for:\n{reasons}"
        )

    def suggest_correction(self) -> str:
        return (
            "If every goal has the same contact, the target pose itself brings the arm "
            "into contact: move the target, or exclude a body the motion is meant to "
            "touch from the planner's collision checks."
        )


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
    The valid alternative positions of the planned joints at the goal that were searched
    for.
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
            "illegal joint states, or check whether the start is itself in collision."
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
