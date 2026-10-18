from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.tasks.joint_tasks import JointWaypointPath

# %% mimic of a runtime planner


@dataclass(eq=False, repr=False)
class FixedWaypointPath(JointWaypointPath):
    """
    A waypoint path whose planner always returns the same waypoints, and remembers where
    the connections were when it was asked to plan.
    """

    waypoints: list[np.ndarray] = field(kw_only=True)
    """
    The waypoints returned on every planning request.
    """

    planned_from: np.ndarray | None = field(init=False, default=None)
    """
    The positions of the connections when the path was planned.
    """

    def plan_waypoints(self, context: MotionStatechartContext) -> list[np.ndarray]:
        self.planned_from = np.array(
            [connection.position for connection in self.connections]
        )
        return self.waypoints
