"""
DAiSy reaches past a box; once collision avoidance holds the reach back, the plan
inserts a recovery before it: the arm retraces the joint states of its last control
cycles and moves its tool above the target, and the reach runs again from there.
Runs in kinematic simulation, so nothing on the network is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from itertools import groupby
from math import pi

import numpy as np

from coraplex.datastructures.enums import InsertionPosition
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.plan_transformation import RecoveryByInsertion, SequentialInsertion
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Parallel, Sequence, StoppedWhenTrue
from cramph.context import ContextExtension, StatechartContext
from cramph.failure_handling import HandledOnFailure
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_statechart.monitors.joint_monitors import RecentJointStates
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import RetraceRecentJointStates
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% the scene


class SceneBody(StrEnum):
    """
    The bodies the demonstration adds to the scene.
    """

    BOX = "box"
    """
    The box standing in the way of the reach.
    """


# %% the recovery


@dataclass(kw_only=True)
class RetraceAndDetourBeforeStuckReach(SequentialInsertion[StoppedWhenTrue]):
    """
    Runs, before the reach its monitor stopped, the retrace of the arm's recent joint
    states and a move of the tool to :attr:`detour_pose`.
    """

    recorder: RecentJointStates
    """
    The monitor that remembered the joint states of the stopped reach.
    """

    kinematic_chain: KinematicChain
    """
    The chain whose tip, the tool frame, is brought to the detour pose.
    """

    detour_pose: Pose
    """
    Where the tool goes after retracing, before the reach runs again.
    """

    detour_position_threshold: float = 0.05
    """
    Distance in meters within which the tool counts as at the detour pose, which only
    has to bring the arm clear of the obstacle.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: StoppedWhenTrue) -> bool:
        return True

    def anchor(self, plan_node: StoppedWhenTrue) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: StoppedWhenTrue) -> list[StatechartNode]:
        return [
            Sequence(
                name="retrace and detour",
                nodes=[
                    RetraceRecentJointStates(name="retrace", monitor=self.recorder),
                    CartesianPose(
                        name="above the target",
                        kinematic_chain=self.kinematic_chain,
                        goal_pose=self.detour_pose,
                        position_threshold=self.detour_position_threshold,
                    ),
                ],
            )
        ]


# %% the demonstration


@dataclass(kw_only=True)
class RetraceDemonstration(RobotDemonstration):
    """
    DAiSy parks its arms and reaches past a box on its table, retracing and detouring
    once the box holds the reach back.
    """

    ros_node_name: str = "daisy_retrace_demo_node"
    """
    The name of the ROS node the demonstration communicates through.
    """

    event_segmentation: bool = False
    """
    Off, since the demonstration shows the plan, not the events in the world.
    """

    collision_avoidance: bool = True
    """
    On, since collision avoidance is what holds the reach back at the box.
    """

    box_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            0.55, 0.39, 0.829
        )
    )
    """
    Where the center of the box stands, at the front edge of the table, between the
    parked right tool and the target.
    """

    box_scale: Scale = field(default_factory=lambda: Scale(0.06, 0.6, 0.442))
    """
    The size of the box.
    """

    target_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            0.75, 0.39, 0.95, roll=pi, yaw=-pi / 2
        )
    )
    """
    Where the right tool reaches to, beyond the box, pointing down.
    """

    target_T_detour: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(z=0.15)
    )
    """
    Where the tool goes after retracing, relative to the target in the frame of the
    world: above the target, clear of the top of the box.
    """

    stall_timeout: timedelta = timedelta(seconds=1.5)
    """
    How long the reach may go without approaching its target before it is stopped.
    """

    retraced_control_cycles: int = 200
    """
    How many of its latest control cycles the arm retraces.
    """

    plan: Sequence | None = field(default=None, init=False)
    """
    The plan performed, None until it was built.
    """

    statechart: Statechart | None = field(default=None, init=False)
    """
    The statechart performing :attr:`plan`, None until it was built.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        :return: A world holding nothing but what :meth:`populate_scene` adds.
        """
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        """
        :return: The DAiSy, standing at the origin of the scene.
        """
        return RobotSpecification(semantic_annotation_type=self.used_robot)

    def build_simulated_world(self) -> World:
        """
        Differs from :meth:`~coraplex.demonstrations.RobotDemonstration.build_simulated_world`
        in starting with the arms parked, since at zero they reach into the grippers.
        """
        world = super().build_simulated_world()
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        return world

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneBody.BOX)

    def populate_scene(self, world: World) -> None:
        """
        Put the box on the table.
        """
        BodySpecification.box(
            SceneBody.BOX, self.box_scale, color=Color(0.8, 0.3, 0.1)
        ).spawn(world, parent_T_self=self.box_pose)

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(
                robot=world.get_semantic_annotations_by_type(self.used_robot)[0]
            )
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Park both arms, then reach past the box with the right tool, recovering once by
        retracing and detouring if the reach stalls.
        """
        world = context.world
        robot = context.require_extension(extension_type=RobotAccess).robot
        kinematic_chain = KinematicChain(
            root=world.root, tip=robot.right_arm.end_effector.tool_frame
        )
        reach = CartesianPose(
            name="reach past the box",
            kinematic_chain=kinematic_chain,
            goal_pose=self._pose_in(world=world, root_T_pose=self.target_pose),
        )
        recorder = RecentJointStates(
            name="recent joint states",
            connections=robot.right_arm.active_connections,
            control_cycles=self.retraced_control_cycles,
        )
        reach_remembering_its_steps = Parallel(
            name="reach remembering its steps",
            nodes=[reach, recorder],
            minimum_success=1,
        )
        self.plan = Sequence(
            nodes=[
                ParkArmsAction(arms=robot.all_arms),
                HandledOnFailure(
                    name="reach, recovering when stuck",
                    handled_node=StoppedWhenTrue(
                        name="reach, stopped when stuck",
                        monitor=Stalled(
                            name="reach stalled",
                            monitored_node=reach_remembering_its_steps,
                            timeout=self.stall_timeout,
                        ),
                        monitored_node=reach_remembering_its_steps,
                    ),
                    failure_handling_strategy=RecoveryByInsertion(
                        maximum_attempts=1,
                        transformation=RetraceAndDetourBeforeStuckReach(
                            recorder=recorder,
                            kinematic_chain=kinematic_chain,
                            detour_pose=self._pose_in(
                                world=world,
                                root_T_pose=self.target_T_detour @ self.target_pose,
                            ),
                        ),
                    ),
                ),
            ]
        )
        self.statechart = Statechart(context=context)
        self.statechart.add_node(node=self.plan)
        return self.statechart

    @staticmethod
    def _pose_in(*, world: World, root_T_pose: HomogeneousTransformationMatrix) -> Pose:
        """
        :return: The pose the transformation moves the root of the world to.
        """
        return HomogeneousTransformationMatrix(
            data=root_T_pose.to_np(), reference_frame=world.root
        ).pose

    def print_what_happened(self) -> None:
        """
        Print the steps the plan ended up with, including the recovery that joined it
        while it ran, and how the reach and the recovery went through their life cycles.
        """
        print("The plan ended up with these steps:")
        for step in self.plan.nodes:
            print(f"  {step.name}: {step.life_cycle_state.name}")
        print("Life cycle of the reach and the recovery:")
        for node in [
            *self.statechart.get_nodes_by_type(node_type=CartesianPose),
            *self.statechart.get_nodes_by_type(node_type=RetraceRecentJointStates),
        ]:
            history = self.statechart.state_history.get_life_cycle_history_of_node(
                node=node
            )
            changes = [life_cycle.name for life_cycle, _ in groupby(history)]
            print(f"  {node.name}: {' -> '.join(changes)}")


def main() -> None:
    """
    Run the demonstration and print what happened.
    """
    demonstration = RetraceDemonstration(used_robot=DAiSy)
    world = demonstration.run()
    demonstration.print_what_happened()
    robot = world.get_semantic_annotations_by_type(DAiSy)[0]
    tool_position = world.compute_forward_kinematics_np(
        world.root, robot.right_arm.end_effector.tool_frame
    )[:3, 3]
    distance = np.linalg.norm(tool_position - demonstration.target_pose.to_np()[:3, 3])
    print(f"The right tool ended {distance * 100:.1f} cm from its target.")


if __name__ == "__main__":
    main()
