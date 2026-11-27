"""
The errors raised while simulating cables with SuperDex.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from semantic_digital_twin.world_description.world_entity import Body

# %% cable simulation


@dataclass(kw_only=True)
class BodyWithoutCollisionShapesError(DataclassException):
    """
    Raised when a body a cable should collide with has no collision shapes to collide
    with.
    """

    body: Body
    """
    The body without collision shapes.
    """

    def error_message(self) -> str:
        return f"The body {self.body.name} has no collision shapes a cable can touch."

    def suggest_correction(self) -> str:
        return "Give the body collision shapes or leave it out of the colliding bodies."


@dataclass(kw_only=True)
class CableSimulationDivergedError(DataclassException):
    """
    Raised when the cable simulation's solver diverged, leaving the cables in a state
    that is no longer physical.
    """

    residual: float
    """
    The residual the solver ended with.
    """

    def error_message(self) -> str:
        return f"The cable simulation diverged with a residual of {self.residual}."

    def suggest_correction(self) -> str:
        return (
            "Step the simulation in shorter periods, shorten the maximum substep duration, or "
            "move the colliding bodies more slowly."
        )


@dataclass(kw_only=True)
class CableNodeOutOfRangeError(DataclassException):
    """
    Raised when a node is addressed that a cable does not have.
    """

    node_index: int
    """
    The index of the node that was addressed.
    """

    number_of_nodes: int
    """
    How many nodes the cable has.
    """

    def error_message(self) -> str:
        return (
            f"The cable has {self.number_of_nodes} nodes; there is no node "
            f"{self.node_index}."
        )

    def suggest_correction(self) -> str:
        return f"Address a node between 0 and {self.number_of_nodes - 1}."
