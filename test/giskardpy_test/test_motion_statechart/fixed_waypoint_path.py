"""
A mimic of a planner of joint waypoint paths, for the tests of the paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from giskardpy.motion_statechart.tasks.joint_tasks import JointWaypointPath

# %% mimic of a runtime planner


@dataclass(eq=False, repr=False, kw_only=True)
class FixedWaypointPath(JointWaypointPath):
    """
    A waypoint path whose planner always returns the same waypoints, and remembers where
    the connections were when it was asked to plan.
    """

    waypoints: list[np.ndarray]
    """
    The waypoints returned on every planning request.
    """

    planned_from: np.ndarray | None = field(init=False, default=None)
    """
    The positions of the connections when the path was planned.
    """

    def plan_waypoints(self) -> list[np.ndarray]:
        self.planned_from = np.array(
            [connection.position for connection in self.connections]
        )
        return self.waypoints
