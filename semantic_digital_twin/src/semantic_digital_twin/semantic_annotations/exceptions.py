from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException

# %% deformable cables


@dataclass
class TooFewCableSegmentsError(DataclassException):
    """
    Raised when a cable is to be made of fewer than one segment.
    """

    number_of_segments: int = field(kw_only=True)
    """
    The number of segments that was asked for.
    """

    def error_message(self) -> str:
        return f"A cable needs at least one segment, not {self.number_of_segments}."

    def suggest_correction(self) -> str:
        return "Ask for one segment or more."


@dataclass
class ArcLengthOutsideCableError(DataclassException):
    """
    Raised when a point is asked for further along a cable than the cable reaches.
    """

    arc_length: float = field(kw_only=True)
    """
    The distance along the cable that was asked for, in metres.
    """

    cable_length: float = field(kw_only=True)
    """
    The length of the cable, in metres.
    """

    def error_message(self) -> str:
        return (
            f"The cable is {self.cable_length} m long; there is no point "
            f"{self.arc_length} m along it."
        )

    def suggest_correction(self) -> str:
        return f"Ask for a distance between 0 and {self.cable_length} m."
