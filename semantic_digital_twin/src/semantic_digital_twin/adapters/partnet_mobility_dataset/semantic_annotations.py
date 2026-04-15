from dataclasses import dataclass

from typing_extensions import ClassVar, Set

from semantic_digital_twin.world_description.world_entity import (
    RootedSemanticAnnotation,
)


@dataclass(eq=False)
class PartNetLabel(RootedSemanticAnnotation):
    """
    Represents a label in the Partnet Mobility dataset semantic annotation hierarchy.
    """

    labels: ClassVar[Set[str]] = set()
    """
    The actual names that represent this in the PartNet Mobility dataset semantics.txt files.
    """
