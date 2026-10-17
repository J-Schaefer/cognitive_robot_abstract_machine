#! /usr/bin/env python3
"""
DAiSy cable demo: home both arms, grasp the cable from its hanger with both hands,
regrasp it above the table and hang it back.

Run it as a script. The classes it defines describe the cable workcell and the plans
of the demo, and are imported by ``experiments.daisy_cable_replanning``, which runs the
demo many times.
"""

# %% Imports

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from math import pi
from pathlib import Path
from time import sleep

# Monorepo imports
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import (
    kinematically_simulated_robot,
    real_robot,
    semi_real_robot,
)
from coraplex.language import SequentialNode
from coraplex.plans.factories import sequential
from coraplex.robot_plans import MoveGripperMotion, MoveJointsMotion
from coraplex.robot_plans.actions.core.cable_actions import (
    CableGraspAction,
    CableRegraspAction,
    CableRehangAction,
)
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

# Custom imports
from define_real_daisy import setup_real_daisy
from define_sim_daisy import setup_sim_daisy
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
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
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Self

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
        cls, world: World, robot: DAiSy, layout: CableWorkcellLayout
    ) -> Self:
        """
        Add the cable to the post and hanger of the world, adding those too where the
        world does not carry them yet, as a world fetched from the real robot may.

        :param world: The world DAiSy stands in.
        :param robot: The robot the post is placed relative to.
        :param layout: Where post, hanger and cable are placed.
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
        mesh_world = STLParser(str(mesh.file_path)).parse()
        mesh_root = mesh_world.root
        with world.modify_world():
            world.merge_world(
                mesh_world,
                FixedConnection(
                    parent, mesh_root, parent_T_connection_expression=parent_T_mesh
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

    context: Context
    """
    The context the plans are performed in.
    """

    workcell: CableWorkcell
    """
    The cable, and the hanger it is taken from and hung back on.
    """

    left_safe_posture: tuple[float, ...] = SafeArmPosture.LEFT.value
    """
    Joint positions the left arm is moved to while homing.
    """

    right_safe_posture: tuple[float, ...] = SafeArmPosture.RIGHT.value
    """
    Joint positions the right arm is moved to before regrasping and rehanging.
    """

    @property
    def robot(self) -> DAiSy:
        """
        :return: The robot performing the demo.
        """
        return self.context.robot

    def for_stage(self, stage: CableDemoStage) -> SequentialNode:
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

    def home(self) -> SequentialNode:
        """
        :return: Cycling both grippers, parking the right arm, moving the left arm to
            its safe posture and half closing both grippers.
        """
        left_gripper = self.robot.left_arm.end_effector
        right_gripper = self.robot.right_arm.end_effector
        return sequential(
            [
                self._both_grippers(
                    left_gripper.default_configuration(GripperState.OPEN),
                    right_gripper.default_configuration(GripperState.OPEN),
                ),
                self._both_grippers(
                    left_gripper.default_configuration(GripperState.CLOSE),
                    right_gripper.default_configuration(GripperState.CLOSE),
                ),
                self._both_grippers(
                    GriplinkPresetConfiguration.from_state_type(
                        left_gripper,
                        GripperState.OPEN,
                        grip_preset=GriplinkGripPreset.PRESET_0,
                    ),
                    GriplinkPresetConfiguration.from_state_type(
                        right_gripper,
                        GripperState.OPEN,
                        grip_preset=GriplinkGripPreset.PRESET_0,
                    ),
                ),
                ParkArmsAction(arms=self.robot.all_arms),
                # TODO Move this motion to the CableGraspAction and add generic version depending on the arm
                self._both_grippers(
                    GriplinkFlexConfiguration.from_state_type(
                        left_gripper,
                        GripperState.FLEXCLOSE,
                        grip_position=70,
                        grip_velocity=300,
                    ),
                    GriplinkFlexConfiguration.from_state_type(
                        right_gripper,
                        GripperState.FLEXCLOSE,
                        grip_position=70,
                        grip_velocity=300,
                    ),
                ),
            ],
            self.context,
        )

    def grasp(self) -> SequentialNode:
        """
        :return: Scooping the cable off the hanger with one arm and grasping it below
            with the other.
        """
        return sequential(
            [
                CableGraspAction(
                    cable_annotation=self.workcell.cable,
                    hanger_body=self.workcell.hanger,
                    grasp_offset=0.1,
                    side_offset=0.2,
                    front_offset=-0.04,
                    down_offset=0.1677,
                    approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
                    approach_sign=-1,  # y-axis pointing to the back
                ),
            ],
            self.context,
        )

    def regrasp(self) -> SequentialNode:
        """
        :return: Moving the right arm out of the way and regrasping the cable with both
            hands above the table.
        """
        return sequential(
            [
                CableRegraspAction(
                    cable_annotation=self.workcell.cable,
                    hanger_body=self.workcell.hanger,
                    approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
                    approach_sign=-1,  # y-axis pointing to the back
                    regrasp_height=0.6,  # height above the table to perform the regrasp
                ),
            ],
            self.context,
        )

    def rehang(self) -> SequentialNode:
        """
        :return: Moving the right arm out of the way and hanging the cable back on the
            hanger.
        """
        return sequential(
            [
                CableRehangAction(
                    cable_annotation=self.workcell.cable,
                    hanger_body=self.workcell.hanger,
                    side_offset=0.05,
                    front_offset=0.07,
                    up_offset=0.03,
                    approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
                    approach_sign=-1,  # y-axis pointing to the back
                ),
            ],
            self.context,
        )

    def _move_right_arm_to_safe_posture(
        self,
    ) -> list[ParkArmsAction | MoveJointsMotion]:
        """
        :return: Parking the right arm and moving it on to its safe posture.
        """
        # TODO: this should be in the action description
        return [
            ParkArmsAction(arms=[self.robot.right_arm]),
            self._move_to_posture(self.robot.right_arm, self.right_safe_posture),
        ]

    @staticmethod
    def _move_to_posture(arm: Arm, posture: tuple[float, ...]) -> MoveJointsMotion:
        """
        :return: The motion moving the arm's joints, from shoulder to wrist, to the
            posture.
        """
        return MoveJointsMotion(
            names=[connection.name.name for connection in arm.active_connections],
            positions=list(posture),
        )

    @staticmethod
    def _both_grippers(
        left_configuration: GripperConfiguration,
        right_configuration: GripperConfiguration,
    ) -> SequentialNode:
        """
        :return: The motions of both WPG/Griplink grippers into their configurations,
            one after the other.
        """
        # shortcut: sequential because parallel() performs each motion in a thread of
        # its own, which races on the shared world in simulation and sends concurrent
        # Giskard goals on the real robot; switch back once ParallelNode is fixed in
        # coraplex.
        return sequential(
            [
                MoveGripperMotion(configuration=left_configuration),
                MoveGripperMotion(configuration=right_configuration),
            ]
        )


# %% Demo

if __name__ == "__main__":
    verbose = True
    collision_avoidance = False
    # execution_mode = ExecutionType.REAL
    execution_mode = ExecutionType.SEMI_REAL
    # execution_mode = ExecutionType.KINEMATICALLY_SIMULATED

    print(f"Running in: {execution_mode}")
    execution_environment = {
        ExecutionType.REAL: real_robot,
        ExecutionType.SEMI_REAL: semi_real_robot,
        ExecutionType.KINEMATICALLY_SIMULATED: kinematically_simulated_robot,
    }[execution_mode]

    # %% Robot and World Setup
    if execution_mode in (ExecutionType.REAL, ExecutionType.SEMI_REAL):
        node, world, robot_view, context = setup_real_daisy()
    else:
        node, world, robot_view, context = setup_sim_daisy()

    # %% Define Additional Objects
    workcell = CableWorkcell.find_or_add_to(world, robot_view, CableWorkcellLayout())
    plans = CableDemoPlans(context=context, workcell=workcell)

    # %% Debug Prints
    if verbose:
        print(world.root.name)
        for dof in context.robot.degrees_of_freedom_with_hardware_interface:
            print(f"{dof.name}: {dof.variables.position.resolve():.2f}")

    # %% Home Robot
    with execution_environment(collision_avoidance=False):
        plans.home().perform()

    sleep(3)

    # %% Grasp Cable
    with execution_environment(collision_avoidance=collision_avoidance):
        plans.grasp().perform()

    # %% Regrasp Cable
    with execution_environment(collision_avoidance=collision_avoidance):
        plans.regrasp().perform()

    # %% Rehang Cable
    with execution_environment(collision_avoidance=collision_avoidance):
        plans.rehang().perform()

    print("Plan finished.")
