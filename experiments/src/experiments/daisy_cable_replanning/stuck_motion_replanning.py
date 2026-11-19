"""
Replanning of stuck pose goals in this process rather than in the controller, so the
statechart of a plan can be sent to a controller running elsewhere, such as the one of
the real robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.context_extensions import RobotAccess
from cramph.composites import (
    CompositeNodeChoosingItsChild,
    ChildChooser,
    StoppedWhenTrue,
    TryInOrder,
)
from cramph.node import StatechartNode
from experiments.daisy_cable_replanning.collisions import IllegalJointStateDatabase
from experiments.daisy_cable_replanning.replanning import (
    MovingArm,
    ReplannedGoal,
    ReplanningLog,
    ReplanningParameters,
    ReplanningTransformation,
)
from giskardpy.motion_statechart.tasks.joint_tasks import FixedJointWaypointPath
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% path planned by the client


@dataclass(eq=False, repr=False, kw_only=True)
class JointPathToPose(CompositeNodeChoosingItsChild):
    """
    Moves an arm's tool frame to a pose along a joint path that the process choosing its
    child plans from where the arm is once this node starts.
    """

    tip_link: KinematicStructureEntity
    """
    The tool frame of the arm that is moved.
    """

    goal_pose: Pose
    """
    Where the tool frame is brought.
    """


@dataclass(kw_only=True)
class JointPathPlanning(ChildChooser[JointPathToPose]):
    """
    Plans the path of a :class:`JointPathToPose` in this process from where the arm is,
    and hands it the path as waypoints followed while Giskard's collision avoidance
    leaves the arm alone; a planned path that failed fails the node.
    """

    parameters: ReplanningParameters = field(default_factory=ReplanningParameters)
    """
    How the path is planned.
    """

    replanning_log: ReplanningLog
    """
    Where the planned paths are recorded.
    """

    def choose_child(self, *, node: JointPathToPose) -> StatechartNode | None:
        if node.latest_child is not None:
            return None
        robot = node.context.require_extension(extension_type=RobotAccess).robot
        moving_arm = MovingArm(
            arm=MovingArm.arm_with_tool_frame(robot, node.tip_link),
            robot=robot,
            world=node.world,
        )
        goals = moving_arm.goal_sampler(self.parameters).configurations_reaching(
            node.goal_pose
        )
        waypoints = moving_arm.path_planner(
            self.parameters,
            IllegalJointStateDatabase.for_daisy_arms().joint_configurations(node.world),
        ).plan(goals)
        self.replanning_log.record(waypoints)
        return moving_arm.without_collision_avoidance(
            FixedJointWaypointPath(
                name="planned joint path",
                connections=moving_arm.arm.active_connections,
                waypoints=[waypoint.tolist() for waypoint in waypoints],
            )
        )


# %% plan transformation


@dataclass
class StuckMotionReplanning(ReplanningTransformation):
    """
    Stops every pose goal once it stops approaching its pose short of reaching it, and
    then reaches the pose along a joint path a :class:`JointPathPlanning` plans.
    """

    def replacement_for(self, replanned: ReplannedGoal) -> TryInOrder:
        goal = replanned.goal
        return TryInOrder(
            name=goal.name,
            nodes=[
                StoppedWhenTrue(monitor=self._stalled(goal), monitored_node=goal),
                JointPathToPose(
                    name="planned joint path to the goal pose",
                    tip_link=goal.tip_link,
                    goal_pose=goal.goal_pose,
                ),
            ],
        )
