#! /usr/bin/env python3

# %% Imports

# General import
import os
import time
from math import pi
from time import sleep
import numpy as np

from coraplex.alternative_motion_mappings.daisy_motion_mapping import (
    DAiSyGripMotion,
    DAiSyFlexGripMotion,
)

# Monorepo imports
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
    WPGGripPreset,
    ExecutionType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot, simulated_robot, semi_real_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans import MoveGripperMotion, MoveJointsMotion
from coraplex.robot_plans.actions.core.cable_grasp import CableGraspAction
from coraplex.robot_plans.actions.core.cable_regrasp import CableRegraspAction
from coraplex.robot_plans.actions.core.cable_rehang import CableRehangAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.orm.ormatic_interface import (
    AllowCollisionForAdjacentPairsDAO,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

# Custom imports
from define_real_daisy import setup_real_daisy
from define_sim_daisy import setup_sim_daisy

verbose = True
collision_avoidance = False
execution_mode = ExecutionType.REAL
# execution_mode = ExecutionType.SEMI_REAL

print(f"Running in: {execution_mode}")

# %% Robot and World Setup
if execution_mode == ExecutionType.REAL or execution_mode == ExecutionType.SEMI_REAL:
    node, world, robot_view, context = setup_real_daisy()
else:
    node, world, robot_view, context = setup_sim_daisy()

# %% Define Additional Objects

# try:
#     cup = world.get_bodies_by_name(PrefixedName("jeroen_cup.stl"))[0]
# except (WorldEntityNotFoundError, IndexError):
#     cup = STLParser(
#         os.path.join(
#             os.path.dirname(__file__),
#             "..",
#             "..",
#             "resources",
#             "objects",
#             "jeroen_cup.stl",
#         )
#     ).parse()
#     cup_root = cup.root
#
#     with world.modify_world():
#         world.merge_world(
#             cup,
#             FixedConnection(
#                 world.root,
#                 cup_root,
#                 parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
#                     -0.6, -0.1, 0.61, reference_frame=world.root
#                 ),
#             ),
#         )

try:
    cable_post = world.get_bodies_by_name(PrefixedName("item_profile_8_40x40_720.stl"))[
        0
    ]
except (WorldEntityNotFoundError, IndexError):
    cable_post = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "item_profile_8_40x40_720.stl",
        )
    ).parse()
    cable_post_root = cable_post.root

    with world.modify_world():
        world.merge_world(
            cable_post,
            FixedConnection(
                world.get_semantic_annotations_by_type(DAiSy)[0].root,
                cable_post_root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.42,
                    y=0.025,
                    z=0.8477,
                    roll=pi / 2,
                    reference_frame=world.get_semantic_annotations_by_type(DAiSy)[
                        0
                    ].root,
                ),
            ),
        )

try:
    cable_hanger = world.get_bodies_by_name(PrefixedName("cable_hanger_2.stl"))[0]
except (WorldEntityNotFoundError, IndexError):
    cable_hanger = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "cable_hanger_2.stl",
        )
    ).parse()
    cable_hanger_root = cable_hanger.root

    with world.modify_world():
        world.merge_world(
            cable_hanger,
            FixedConnection(
                cable_post_root,
                cable_hanger_root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.0,
                    y=0.310,  # 720/2 - 50
                    z=0.02,
                    roll=-pi / 2,
                    # pitch=-pi / 2,
                    reference_frame=cable_post_root,
                ),
            ),
        )

hanger_body = world.get_body_by_name(PrefixedName("cable_hanger_2.stl"))

# %% Cable Definition
# TODO: Figure out how to respawn the cable at a new position
with world.modify_world():
    cable_annotation = Cable.create_with_new_body_in_world(
        name=PrefixedName("cable"),
        world=world,
        hanging_from=hanger_body,
        length=0.3,
        mount_offset_x=0.055,
        mount_offset_y=-0.07,
        height_offset=0.0,
    )

    world.collision_manager.extend_default_rules(
        [AllowCollisionForBodies(allowed_collision_bodies={cable_annotation.root})]
    )

# %% Debug Prints
if verbose:
    print(world.root.name)

    # Print joint states
    for dof in context.robot.degrees_of_freedom_with_hardware_interface:
        print(f"{dof.name}: {dof.variables.position.resolve():.2f}")

# %% Home Robot

daisy_left_arm_names = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

daisy_safe_left_arm_positions = [
    -2.71,  # left_shoulder_pan_joint
    -1.01,  # left_shoulder_lift_joint
    -2.10,  # left_elbow_joint
    -1.59,  # left_wrist_1_joint
    1.53,  # left_wrist_2_joint
    -4.23,  # left_wrist_3_joint
]

daisy_right_arm_names = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]

# daisy_safe_right_arm_positions = [
#     2.17,  # right_shoulder_pan_joint
#     -2.17,  # right_shoulder_lift_joint
#     2.04,  # right_elbow_joint
#     -1.43,  # right_wrist_1_joint
#     -1.59,  # right_wrist_2_joint
#     1.39,  # right_wrist_3_joint
# ]
daisy_safe_right_arm_positions = [
    -0.02,  # right_shoulder_pan_joint
    -0.97,  # right_shoulder_lift_joint
    -2.00,  # right_elbow_joint
    -1.76,  # right_wrist_1_joint
    1.56,  # right_wrist_2_joint
    -0.83,  # right_wrist_3_joint
]


plan_home = sequential(
    [
        MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.BOTH),
        MoveGripperMotion(motion=GripperState.CLOSE, gripper=Arms.BOTH),
        DAiSyGripMotion(motion=GripperState.OPEN, gripper=Arms.BOTH),
        ParkArmsAction(arm=Arms.RIGHT),
        MoveJointsMotion(  # TODO Move this motion to the CableGraspAction and add generic version depending on the arm
            names=daisy_left_arm_names, positions=daisy_safe_left_arm_positions
        ),
        DAiSyFlexGripMotion(
            motion=GripperState.FLEXCLOSE,
            gripper=Arms.BOTH,
            grip_position=70,
            grip_speed=300,
        ),
        DAiSyGripMotion(
            motion=GripperState.OPEN,
            gripper=Arms.BOTH,
            grip_preset=WPGGripPreset.PRESET_0,
        ),
    ],
    context,
)

if execution_mode == ExecutionType.REAL:
    with real_robot(collision_avoidance=False):
        plan_home.perform()
elif execution_mode == ExecutionType.SEMI_REAL:
    with semi_real_robot(collision_avoidance=False):
        plan_home.perform()
else:
    with simulated_robot:
        plan_home.perform()

sleep(3)

# %% Demo Plan
pick_up_grasp = GraspDescription(
    approach_direction=ApproachDirection.LEFT,
    vertical_alignment=VerticalAlignment.NoAlignment,
    end_effector=context.robot.get_right_arm_if_specified().end_effector,
    manipulation_offset=0.05,
)

plan = sequential(
    [
        # PickUpAction(
        #     object_designator=cup_root, arm=Arms.RIGHT, grasp_description=pick_up_grasp
        # ),
        CableGraspAction(
            cable_annotation=cable_annotation,
            hanger_body=hanger_body,
            grasp_offset=0.1,
            side_offset=0.2,
            front_offset=-0.04,
            down_offset=0.1677,
            approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
            approach_sign=-1,  # y-axis pointing to the back
        ),
    ],
    context,
)

if execution_mode == ExecutionType.REAL:
    with real_robot(collision_avoidance=collision_avoidance):
        plan.perform()
elif execution_mode == ExecutionType.SEMI_REAL:
    with semi_real_robot(collision_avoidance=collision_avoidance):
        plan.perform()
else:
    with simulated_robot:
        plan.perform()

plan_regrasp = sequential(
    [
        ParkArmsAction(
            arm=Arms.RIGHT
        ),  # TODO: this should be in the action description
        MoveJointsMotion(  # TODO Move this motion to the CableGraspAction and add generic version depending on the arm
            names=daisy_right_arm_names, positions=daisy_safe_right_arm_positions
        ),
        CableRegraspAction(
            cable_annotation=cable_annotation,
            hanger_body=hanger_body,
            approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
            approach_sign=-1,  # y-axis pointing to the back
            regrasp_height=0.6,  # height above the table to perform the regrasp
        ),
        # ParkArmsAction(arm=Arms.BOTH),
    ],
    context,
)

if execution_mode == ExecutionType.REAL:
    with real_robot(collision_avoidance=collision_avoidance):
        plan_regrasp.perform()
elif execution_mode == ExecutionType.SEMI_REAL:
    with semi_real_robot(collision_avoidance=collision_avoidance):
        plan_regrasp.perform()
else:
    with simulated_robot:
        plan_regrasp.perform()

# %% Rehang Cable
plan_rehang = sequential(
    [
        CableRehangAction(
            cable_annotation=cable_annotation,
            hanger_body=hanger_body,
            side_offset=0.05,
            front_offset=0.07,
            up_offset=0.03,
            approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
            approach_sign=-1,  # y-axis pointing to the back
        )
    ],
    context,
)

if execution_mode == ExecutionType.REAL:
    with real_robot(collision_avoidance=collision_avoidance):
        plan_rehang.perform()
elif execution_mode == ExecutionType.SEMI_REAL:
    with semi_real_robot(collision_avoidance=collision_avoidance):
        plan_rehang.perform()
else:
    with simulated_robot:
        plan_rehang.perform()

print("Plan finished.")
