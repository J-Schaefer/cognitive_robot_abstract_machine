#! /usr/bin/env python3
"""
DAiSy grasps a cable from its hanger with both arms, regrasps it above the table and
hangs it back onto the hanger.

Run it as a script and choose where the plan runs with ``executor_type``:

- :class:`~coraplex.plans.executors.KinematicallySimulatedPlanExecutor` simulates DAiSy
  in this process,
- :class:`~coraplex.plans.executors.RealRobotPlanExecutor` sends the plan to a running
  standalone controller (``ros2 launch giskardpy_ros giskardpy_daisy_standalone.launch.py``),
- :class:`~coraplex.griplink_grippers.RealDAiSyPlanExecutor` drives the real DAiSy and
  its griplink grippers.

Every stage is performed on its own, since each cable action computes its poses from the
world as the previous stage left it. The classes it defines describe the cable workcell
and the plans of the demo, and are imported by ``experiments.daisy_cable_replanning``,
which runs the demo many times.
"""

# %% Imports

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from math import pi
from pathlib import Path
from time import sleep

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.griplink_grippers import RealDAiSyPlanExecutor  # noqa: F401
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (  # noqa: F401
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
    SimulatedPlanExecutor,
)
from coraplex.robot_plans.actions.core.cable_actions import (
    CableGraspAction,
    CableRegraspAction,
    CableRehangAction,
)
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelReloadSynchronizer,
    WorldSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkFlexConfiguration,
    GriplinkGripPreset,
    GriplinkPresetConfiguration,
)
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Callable, Self

# %% Cable Workcell Definition


class CableWorkcellMesh(Enum):
    """
    Meshes of the cable workcell, as files in the object resources of coraplex.
    """

    CABLE_POST = Path("item_profile_8_40x40_720.stl")
    CABLE_HANGER = Path("cable_hanger_2.stl")

    @property
    def file_path(self) -> Path:
        """
        :return: The absolute path of the mesh file.
        """
        coraplex_root = Path(__file__).resolve().parents[2]
        return coraplex_root / "resources" / "objects" / self.value

    @property
    def body_name(self) -> PrefixedName:
        """
        :return: The name the parsed mesh body carries in the world.
        """
        return PrefixedName(self.value.name)


@dataclass(frozen=True)
class CableWorkcellLayout:
    """
    Where the cable post stands next to DAiSy, where the hanger sits on it and where the
    cable hangs from the hanger.
    """

    post_x: float = 0.625
    """
    Position of the post along the x axis of the robot root, in metres.
    """

    post_y: float = 0.42
    """
    Position of the post along the y axis of the robot root, in metres.
    """

    post_z: float = 0.8477
    """
    Height of the post origin above the robot root, in metres.
    """

    post_roll: float = pi / 2
    """
    Roll of the post relative to the robot root, in radians.
    """

    post_yaw: float = -pi / 2
    """
    Yaw of the post relative to the robot root, in radians.
    """

    hanger_height_on_post: float = 0.310
    """
    Distance of the hanger from the post origin along the post, in metres (720 / 2 -
    50 mm).
    """

    hanger_distance_from_post: float = 0.02
    """
    Distance of the hanger origin from the post axis, in metres.
    """

    cable_length: float = 0.3
    """
    Length of the cable, in metres.
    """

    cable_mount_offset_x: float = 0.055
    """
    Offset of the hanging point along the x axis of the hanger, in metres.
    """

    cable_mount_offset_y: float = -0.07
    """
    Offset of the hanging point along the y axis of the hanger, in metres.
    """


@dataclass
class CableWorkcell:
    """
    The cable post next to DAiSy, the hanger mounted on it and the cable hanging from the
    hanger.
    """

    cable: Cable
    """
    The cable the demo grasps, regrasps and hangs back.
    """

    hanger: Body
    """
    The hanger the cable hangs from.
    """

    @classmethod
    def find_or_add_to(
        cls,
        world: World,
        robot: DAiSy,
        layout: CableWorkcellLayout,
        arm_buffer_zone_distance: float = 0.002,
    ) -> Self:
        """
        Add the cable to the post and hanger of the world, adding those too where the
        world does not carry them yet, as a world fetched from the real robot may.

        :param world: The world DAiSy stands in.
        :param robot: The robot the post is placed relative to.
        :param layout: Where post, hanger and cable are placed.
        :param arm_buffer_zone_distance: Distance in metres collision avoidance keeps
            the moving bodies of the arms from the post and the hanger, small enough for
            the scoop to put the fingers around the cable next to the post.
        :return: The workcell in the world.
        """
        posts = world.get_bodies_by_name(CableWorkcellMesh.CABLE_POST.body_name)
        post = posts[0] if posts else cls._add_post(world, robot, layout)
        hangers = world.get_bodies_by_name(CableWorkcellMesh.CABLE_HANGER.body_name)
        hanger = hangers[0] if hangers else cls._add_hanger(world, post, layout)
        # TODO: Figure out how to respawn the cable at a new position
        with world.modify_world():
            cable = Cable.create_with_new_body_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=hanger,
                length=layout.cable_length,
                mount_offset_x=layout.cable_mount_offset_x,
                mount_offset_y=layout.cable_mount_offset_y,
            )
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionForBodies(allowed_collision_bodies={cable.root})
            )
            world.collision_manager.extend_default_rules(
                [
                    AvoidCollisionBetweenGroups(
                        buffer_zone_distance=arm_buffer_zone_distance,
                        violated_distance=0.0,
                        body_group_a=[
                            body
                            for arm in robot.all_arms
                            for body in world.get_kinematic_structure_entities_of_branch(
                                arm.active_connections[0].child
                            )
                            if isinstance(body, Body) and body.has_collision()
                        ],
                        body_group_b=[post, hanger],
                    )
                ]
            )
        return cls(cable=cable, hanger=hanger)

    @staticmethod
    def _add_post(world: World, robot: DAiSy, layout: CableWorkcellLayout) -> Body:
        """
        :return: The post, newly fixed next to the robot.
        """
        return CableWorkcell._merge_mesh(
            world,
            CableWorkcellMesh.CABLE_POST,
            robot.root,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=layout.post_x,
                y=layout.post_y,
                z=layout.post_z,
                roll=layout.post_roll,
                yaw=layout.post_yaw,
                reference_frame=robot.root,
            ),
        )

    @staticmethod
    def _add_hanger(world: World, post: Body, layout: CableWorkcellLayout) -> Body:
        """
        :return: The hanger, newly fixed to the post.
        """
        return CableWorkcell._merge_mesh(
            world,
            CableWorkcellMesh.CABLE_HANGER,
            post,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                y=layout.hanger_height_on_post,
                z=layout.hanger_distance_from_post,
                roll=-pi / 2,
                reference_frame=post,
            ),
        )

    @staticmethod
    def _merge_mesh(
        world: World,
        mesh: CableWorkcellMesh,
        parent: Body,
        parent_T_mesh: HomogeneousTransformationMatrix,
    ) -> Body:
        """
        :return: The root body of the mesh, newly fixed to the parent.
        """
        mesh_world = STLParser(file_path=str(mesh.file_path)).create_world()
        mesh_root = mesh_world.root
        with world.modify_world():
            world.merge_world(
                mesh_world,
                FixedConnection(
                    parent=parent,
                    child=mesh_root,
                    parent_T_connection_expression=parent_T_mesh,
                ),
            )
        return mesh_root


# %% Safe Arm Postures


class SafeArmPosture(Enum):
    """
    Joint positions, from shoulder pan to wrist 3, an arm is moved to before the other
    arm starts working near the hanger.
    """

    LEFT = (-2.71, -1.01, -2.10, -1.59, 1.53, -4.23)
    RIGHT = (-0.02, -0.97, -2.00, -1.76, 1.56, -0.83)


# %% Demo Plans


class CableDemoStage(StrEnum):
    """
    The stages of the cable demo, in the order they are performed.
    """

    HOME = "home"
    GRASP = "grasp"
    REGRASP = "regrasp"
    REHANG = "rehang"


@dataclass
class CableDemoPlans:
    """
    The plans of the cable demo, one per :class:`CableDemoStage`.
    """

    robot: DAiSy
    """
    The robot performing the demo.
    """

    workcell: CableWorkcell
    """
    The cable, and the hanger it is taken from and hung back on.
    """

    def for_stage(self, stage: CableDemoStage) -> StatechartNode:
        """
        :return: The plan of the stage.
        """
        match stage:
            case CableDemoStage.HOME:
                return self.home()
            case CableDemoStage.GRASP:
                return self.grasp()
            case CableDemoStage.REGRASP:
                return self.regrasp()
            case CableDemoStage.REHANG:
                return self.rehang()

    def home(self) -> Sequence:
        """
        :return: Cycling both grippers, parking both arms and half closing both
            grippers.
        """
        return Sequence(
            nodes=[
                self._both_grippers(
                    lambda gripper: gripper.default_configuration(GripperState.OPEN)
                ),
                self._both_grippers(
                    lambda gripper: gripper.default_configuration(GripperState.CLOSE)
                ),
                self._both_grippers(
                    lambda gripper: GriplinkPresetConfiguration.from_state_type(
                        gripper,
                        GripperState.OPEN,
                        grip_preset=GriplinkGripPreset.PRESET_0,
                    )
                ),
                ParkArmsAction(arms=self.robot.all_arms),
                # TODO Move this motion to the CableGraspAction and add generic version depending on the arm
                self._both_grippers(
                    lambda gripper: GriplinkFlexConfiguration.from_state_type(
                        gripper,
                        GripperState.FLEXCLOSE,
                        grip_position=0.07,
                        grip_velocity=0.3,
                    )
                ),
            ]
        )

    def grasp(self) -> CableGraspAction:
        """
        :return: Scooping the cable off the hanger with one arm and grasping it below
            with the other.
        """
        return CableGraspAction(
            cable_annotation=self.workcell.cable,
            hanger_body=self.workcell.hanger,
            grasp_offset=0.1,
            side_offset=0.2,
            front_offset=-0.04,
            down_offset=0.1677,
            approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
            approach_sign=-1,  # y-axis pointing to the back
        )

    def regrasp(self) -> CableRegraspAction:
        """
        :return: Regrasping the cable with both hands above the table.
        """
        return CableRegraspAction(
            cable_annotation=self.workcell.cable,
            hanger_body=self.workcell.hanger,
            approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
            approach_sign=-1,  # y-axis pointing to the back
            regrasp_height=0.6,  # height above the table to perform the regrasp
        )

    def rehang(self) -> CableRehangAction:
        """
        :return: Hanging the cable back on the hanger.
        """
        return CableRehangAction(
            cable_annotation=self.workcell.cable,
            hanger_body=self.workcell.hanger,
            side_offset=0.05,
            front_offset=0.07,
            up_offset=0.03,
            approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
            approach_sign=-1,  # y-axis pointing to the back
        )

    def _both_grippers(
        self, configuration_of: Callable[[EndEffector], GripperConfiguration]
    ) -> Sequence:
        """
        :param configuration_of: The configuration to move a gripper into.
        :return: Moving the left, then the right griplink gripper, one after the other
            since moving them in parallel fails on the real robot.
        """
        return Sequence(
            nodes=[
                SetGripperAction(configuration=configuration_of(arm.end_effector))
                for arm in (self.robot.left_arm, self.robot.right_arm)
            ]
        )


# %% Demo

if __name__ == "__main__":
    verbose = True
    collision_avoidance = True
    executor_type: type[PlanExecutor] = RealRobotPlanExecutor
    # executor_type = RealDAiSyPlanExecutor
    # executor_type = KinematicallySimulatedPlanExecutor

    print(f"Running with: {executor_type.__name__}")

    # %% Robot and World Setup
    ros_session = RobotDemonstrationRosSession.start("daisy_cable_demo_node")
    if issubclass(executor_type, SimulatedPlanExecutor):
        world = URDFParser.from_file(file_path=DAiSy.get_ros_file_path()).create_world()
        DAiSy.from_world(world)
        VizMarkerPublisher(_world=world, node=ros_session.node)
    else:
        world = ros_session.fetch_world()
        WorldSynchronizer(_world=world, node=ros_session.node)
        ModelReloadSynchronizer(_world=world, node=ros_session.node)

    robot_view = world.get_semantic_annotations_by_type(DAiSy)[0]
    workcell = CableWorkcell.find_or_add_to(world, robot_view, CableWorkcellLayout())
    plans = CableDemoPlans(robot=robot_view, workcell=workcell)

    # %% Debug Prints
    if verbose:
        print(world.root.name)
        for dof in robot_view.degrees_of_freedom_with_hardware_interface:
            print(f"{dof.name}: {dof.variables.position.resolve():.2f}")

    # %% Demo Plan
    for stage in CableDemoStage:
        executor = executor_type(
            context=StatechartContext.with_extensions(
                world=world, extensions=[RobotAccess(robot=robot_view)]
            ),
            ros_node=ros_session.node,
            collision_avoidance=collision_avoidance
            and stage is not CableDemoStage.HOME,
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(node=plans.for_stage(stage))
        try:
            executor.compile(statechart=statechart)
            executor.execute()
        finally:
            executor.close()
        if stage is CableDemoStage.HOME:
            sleep(3)

    print("Plan finished.")
    ros_session.stop()
