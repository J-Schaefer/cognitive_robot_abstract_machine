from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

from krrood.utils import DataclassException

if TYPE_CHECKING:
    from pycram.plans.designator import Designator


@dataclass
class PlanNodeIsNone(DataclassException):
    """
    Raised when an instance managed by a plan node has no plan node.

    Most likely raised when an action created a subplan without calling `ActionDescription.add_subplan`
    """

    instance: Designator
    """
    The instance where the plan node is None.
    """

    def __post_init__(self):
        self.message = (
            f"{self.instance} has no plan node. Did you forget to call `add_subplan` when creating"
            f"plans inside actions?"
        )
