from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from coraplex.robot_plans.actions.core.misc import DetectAction
    from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
    from semantic_digital_twin.world_description.world_entity import Body

# %% detecting


@dataclass
class DetectionTargetMissing(DataclassException):
    """
    Raised when a detection is given neither the type of object to look for nor the box
    to look in.
    """

    instance: DetectAction
    """
    The detection that has no target.
    """

    def error_message(self) -> str:
        return f"{self.instance} has neither an object type nor a detection box."

    def suggest_correction(self) -> str:
        return "provide the type of object to detect, a box to detect in, or both."


# %% sending the fingers to a width


@dataclass
class WidthBeyondGripperOpening(DataclassException):
    """
    Raised when the fingers of a gripper are sent to a width wider than the gripper
    opens.
    """

    gripper: HasTwoFingers
    """
    The gripper whose fingers were sent.
    """

    width: float
    """
    How far apart the fingertips' facing surfaces were sent to stand, in metres.
    """

    def error_message(self) -> str:
        return (
            f"{self.gripper.name} opens only {self.gripper.separation_when_open:.3f} m "
            f"wide, but its fingers were sent {self.width:.3f} m apart."
        )

    def suggest_correction(self) -> str:
        return (
            "grasp the object across a side narrower than the gripper opens, or with a "
            "gripper that opens wider."
        )


# %% handling a cable


@dataclass
class CableNotHeld(DataclassException):
    """
    Raised when an action needs the arm holding a cable, but no end effector holds it.
    """

    cable_body: Body
    """
    The body of the cable that is not held.
    """

    def error_message(self) -> str:
        return f"{self.cable_body.name} is not attached to any end effector."

    def suggest_correction(self) -> str:
        return "perform the cable grasp before an action that needs the cable held."
