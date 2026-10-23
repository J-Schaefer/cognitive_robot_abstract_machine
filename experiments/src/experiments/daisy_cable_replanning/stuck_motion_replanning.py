"""
Replanning of stuck tool center point motions in this process rather than in the
controller, so the motion charts can be sent to a controller running elsewhere, such as
the one of the real robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from coraplex.datastructures.enums import MovementType
from coraplex.plans.executables import Executable
from coraplex.plans.plan_node import ActionNode, ExecutionBoundaryNode, MotionNode
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from experiments.daisy_cable_replanning.collisions import IllegalJointStateDatabase
from experiments.daisy_cable_replanning.replanning import (
    ReplanningTransformation,
    ToolCenterPointMotionWithReplanning,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.payload_monitors import Print
from giskardpy.motion_statechart.monitors.progress_monitors import NoLongerProgressing
from giskardpy.motion_statechart.monitors.templates import ReplacedWhenTrue
from giskardpy.motion_statechart.tasks.joint_tasks import FixedJointWaypointPath

# %% motions


@dataclass
class ToolCenterPointMotionEndingWhenStuck(ToolCenterPointMotionWithReplanning):
    """
    Moves the tool center point like :class:`MoveToolCenterPointMotion`, but ends
    without failing once the motion stops approaching its target short of reaching it.
    """

    @property
    def reached_target(self) -> bool:
        """
        :return: Whether the tool frame is at the target within the motion's
            thresholds; a translation ignores the orientation.
        """
        tool_frame_pose = self.world.transform(
            self.arm.end_effector.tool_frame.global_pose, self.target.reference_frame
        )
        position_error = tool_frame_pose.position.euclidean_distance(
            self.target.position
        ).to_np()
        if position_error > self.resolved_position_threshold():
            return False
        if self.movement_type == MovementType.TRANSLATION:
            return True
        orientation_error = tool_frame_pose.rotation_matrix.rotational_distance(
            self.target.rotation_matrix
        ).to_np()
        return orientation_error <= self.resolved_orientation_threshold()

    @property
    def _motion_chart(self) -> ReplacedWhenTrue:
        motion = super()._motion_chart
        return ReplacedWhenTrue(
            name="MoveTCP or stuck report",
            monitor=NoLongerProgressing(
                name="no longer progressing",
                monitored_node=motion,
                timeout=timedelta(seconds=self.parameters.stagnation_timeout),
            ),
            monitored_node=motion,
            # Ends the chart without failing; the path is planned outside of it.
            replacement=Print(
                name="stuck report",
                message=f"{motion.name} got stuck; a joint path is planned to its "
                "target.",
            ),
        )


@dataclass
class PlannedToolCenterPointMotion(ToolCenterPointMotionWithReplanning):
    """
    Moves the tool center point to its target along a collision-free joint path of the
    arm, planned in this process when the motion chart is built.
    """

    @property
    def _motion_chart(self) -> Sequence:
        goals = self._goal_sampler.configurations_reaching(self.target)
        waypoints = self._planner(
            IllegalJointStateDatabase.for_daisy_arms().joint_configurations(self.world)
        ).plan(goals)
        self.replanning_log.record(waypoints)
        return self._without_collision_avoidance_of_arm(
            FixedJointWaypointPath(
                name="planned joint path",
                connections=self.arm.active_connections,
                waypoints=[waypoint.tolist() for waypoint in waypoints],
            )
        )


# %% plan nodes


@dataclass(eq=False, repr=False)
class PlannedPathAfterStuckMotion(ExecutionBoundaryNode):
    """
    Performs its child, a :class:`PlannedToolCenterPointMotion`, only if the motion
    before it ended short of its target.
    """

    stuck_motion: ToolCenterPointMotionEndingWhenStuck = field(kw_only=True)
    """
    The motion whose getting stuck the planned path follows.
    """

    def parse(self) -> PlannedPathAfterStuckMotionExecutable:
        return PlannedPathAfterStuckMotionExecutable(
            context=self.context, stuck_motion=self.stuck_motion, path=self.children[0]
        )


@dataclass
class PlannedPathAfterStuckMotionExecutable(Executable):
    """
    Performs the planned path once the motion before it has run, if it ended short of
    its target.
    """

    stuck_motion: ToolCenterPointMotionEndingWhenStuck = field(kw_only=True)
    """
    The motion whose getting stuck the planned path follows.
    """

    path: MotionNode = field(kw_only=True)
    """
    The node of the :class:`PlannedToolCenterPointMotion`.
    """

    def execute(self) -> None:
        if self.stuck_motion.reached_target:
            return
        self.path.perform()


# %% plan transformation


@dataclass
class StuckMotionReplanning(ReplanningTransformation):
    """
    Lets every tool center point motion end once it gets stuck, and follows it with a
    path to its target, planned in this process if it got stuck.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return True

    def replacement_for(
        self, motion: MoveToolCenterPointMotion
    ) -> ToolCenterPointMotionEndingWhenStuck:
        return ToolCenterPointMotionEndingWhenStuck.replacing(
            motion, parameters=self.parameters, replanning_log=self.replanning_log
        )

    def rewrite_motion_node(self, node: MotionNode) -> None:
        original_motion = node.designator
        super().rewrite_motion_node(node)
        path_node = PlannedPathAfterStuckMotion(stuck_motion=node.designator)
        node.plan.insert_after(node, path_node)
        node.plan.insert_as_last_child(
            path_node,
            MotionNode(
                designator=PlannedToolCenterPointMotion.replacing(
                    original_motion,
                    parameters=self.parameters,
                    replanning_log=self.replanning_log,
                )
            ),
        )
