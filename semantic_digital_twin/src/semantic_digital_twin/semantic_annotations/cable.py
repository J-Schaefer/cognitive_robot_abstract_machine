from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class Cable(HasRootBody):
    """
    A cable hanging from a fixture such as a cable hanger.
    """

    hanging_from: Body
    """
    The body from which this cable hangs.
    """

    length: float
    """
    The length of the cable in metres.
    """
