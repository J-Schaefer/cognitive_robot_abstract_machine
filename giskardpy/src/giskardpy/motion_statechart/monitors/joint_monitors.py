from __future__ import annotations

from dataclasses import field, dataclass

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts


@dataclass(eq=False, repr=False)
class JointPositionReached(MotionStatechartNode):
    """
    Monitors if a joint position is reached within a certain threshold.
    """

    connection: ActiveConnection1DOF = field(kw_only=True)
    """
    Monitored joint connection.
    """

    position: float = field(kw_only=True)
    """
    Target position to monitor.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    Threshold for position error.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        current = self.connection.dof.variables.position
        if (
            isinstance(self.connection, RevoluteConnection)
            and not self.connection.dof.has_position_limits()
        ):
            error = sm.shortest_angular_distance(current, self.position)
        else:
            error = self.position - current
        return NodeArtifacts(observation=sm.abs(error) < self.threshold)


# %% closeness to a set of joint configurations


@dataclass
class JointConfigurationSet:
    """
    Joint configurations of a group of connections, each compared to other
    configurations only on the connections it marks as relevant.
    """

    connections: list[ActiveConnection1DOF]
    """
    The connections the configurations give positions for, in column order.
    """

    positions: np.ndarray
    """
    One configuration per row, one connection per column.
    """

    relevant_connections: np.ndarray
    """
    Per configuration and connection, whether the connection's position counts when the
    configuration is compared to another one.
    """

    def distance_to(self, positions: np.ndarray) -> float:
        """
        :param positions: Positions of :attr:`connections`.
        :return: The smallest, over all configurations, largest position difference on
            the connections the configuration marks as relevant; infinite if the set is
            empty.
        """
        if len(self.positions) == 0:
            return np.inf
        differences = np.where(
            self.relevant_connections, np.abs(self.positions - positions), 0.0
        )
        return float(np.min(np.max(differences, axis=1)))

    def current_positions(self) -> np.ndarray:
        """
        :return: The positions the connections are at in the world.
        """
        return np.array([connection.position for connection in self.connections])


@dataclass(eq=False, repr=False)
class CloseToJointConfigurations(MotionStatechartNode):
    """
    Observes True while the connections are closer than :attr:`threshold` to a
    configuration of a set, and remembers the last positions they held while at least
    :attr:`safe_distance` away from all of them.
    """

    configurations: JointConfigurationSet = field(kw_only=True)
    """
    The configurations to stay away from.
    """

    threshold: float = field(default=0.15, kw_only=True)
    """
    Distance, as measured by :meth:`JointConfigurationSet.distance_to`, below which the
    connections count as close.
    """

    safe_distance: float = field(default=0.3, kw_only=True)
    """
    Distance at or above which positions count as safe.
    """

    last_safe_positions: np.ndarray | None = field(init=False, default=None)
    """
    The last positions of the connections that were at least :attr:`safe_distance` away
    from every configuration, or the positions this node started at if there were none.
    """

    def on_start(self, context: MotionStatechartContext):
        self.last_safe_positions = self.configurations.current_positions()

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        positions = self.configurations.current_positions()
        distance = self.configurations.distance_to(positions)
        if distance >= self.safe_distance:
            self.last_safe_positions = positions
        if distance < self.threshold:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
