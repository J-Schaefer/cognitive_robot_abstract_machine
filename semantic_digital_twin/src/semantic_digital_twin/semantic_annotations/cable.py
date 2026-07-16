from __future__ import annotations

from dataclasses import dataclass, field

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

    mount_offset_x: float = field(default=0.0)
    """
    Offset in metres along the parent body's local X axis for the hanging point.
    """

    mount_offset_y: float = field(default=0.0)
    """
    Offset in metres along the parent body's local Y axis for the hanging point.
    """

    height_offset: float = field(default=0.0)
    """
    Offset in metres along the parent body's local Z axis from the parent origin down to
    the hanging point.
    """
