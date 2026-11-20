"""
Monitors of the positions of joints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
from semantic_digital_twin.world_description.connections import (
    ScrewAxisConnection,
    ActiveConnection1DOF,
)
from cramph.data_types import ObservationStateValue
from cramph.node import EndedByOwner
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.mixins import HasThresholdParameter
from cramph.node import NodeArtifacts

# %% joint monitors


@dataclass(eq=False, repr=False, kw_only=True)
class JointPositionReached(EndedByOwner, MotionStatechartNode, HasThresholdParameter):
    """
    Monitors if a joint position is reached within a certain threshold.
    """

    connection: ActiveConnection1DOF
    """
    Monitored joint connection.
    """

    position: float
    """
    Target position to monitor.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        :return: The observation that the connection is within the threshold of the
            target position, measured along the shortest angle for a revolute connection
            without position limits.
        """
        current = self.connection.dof.variables.position
        if (
            isinstance(self.connection, ScrewAxisConnection)
            and self.connection.behaves_like_a_revolute_connection
        ) and not self.connection.dof.has_position_limits():
            error = sm.shortest_angular_distance(current, self.position)
        else:
            error = self.position - current
        return NodeArtifacts(observation=sm.abs(error) < self.threshold)


# %% recent joint states


@dataclass(eq=False, repr=False, kw_only=True)
class RecentJointStates(EndedByOwner, MotionStatechartNode):
    """
    Remembers the positions the connections held in the last :attr:`control_cycles`
    control cycles while it runs; observes False throughout.
    """

    connections: list[ActiveConnection1DOF]
    """
    The connections whose positions are remembered.
    """

    control_cycles: int = 200
    """
    How many of the latest positions are remembered.
    """

    states: deque[np.ndarray] = field(init=False, default_factory=deque)
    """
    The remembered positions of :attr:`connections`, one per control cycle, the earliest
    first.
    """

    def on_start(self) -> None:
        """
        Differs from :meth:`~cramph.node.StatechartNode.on_start` in remembering only
        the positions the connections start at.
        """
        self.states = deque([self._current_positions()], maxlen=self.control_cycles)

    def on_tick(self) -> ObservationStateValue:
        """
        Differs from :meth:`~cramph.node.StatechartNode.on_tick` in remembering the
        positions of this control cycle.
        """
        self.states.append(self._current_positions())
        return ObservationStateValue.FALSE

    def _current_positions(self) -> np.ndarray:
        """
        :return: The positions :attr:`connections` are at.
        """
        return np.array([connection.position for connection in self.connections])
