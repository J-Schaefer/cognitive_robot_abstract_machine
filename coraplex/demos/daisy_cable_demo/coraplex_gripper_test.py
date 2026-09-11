#! /usr/bin/env python3

# %% Imports

# General import
import os
import time
from math import pi
from time import sleep
import numpy as np

# Monorepo imports
from coraplex.datastructures.enums import (
    Arms,
    ExecutionType,
)
from coraplex.execution_environment import real_robot, simulated_robot, semi_real_robot
from coraplex.plans.factories import sequential, parallel
from coraplex.robot_plans import MoveGripperMotion, MoveJointsMotion
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import DAiSy

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

daisy_safe_right_arm_positions = [
    -0.02,  # right_shoulder_pan_joint
    -0.97,  # right_shoulder_lift_joint
    -2.00,  # right_elbow_joint
    -1.76,  # right_wrist_1_joint
    1.56,  # right_wrist_2_joint
    -0.83,  # right_wrist_3_joint
]

daisy = world.get_semantic_annotations_by_type(DAiSy)[0]
left_gripper = daisy.left_arm.end_effector
right_gripper = daisy.right_arm.end_effector
open_spec_l = left_gripper.default_specification(GripperState.OPEN)
close_spec_l = left_gripper.default_specification(GripperState.CLOSE)
flex_open_spec_l = left_gripper.default_specification(
    GripperState.FLEXOPEN,
)
flex_close_spec_l = left_gripper.default_specification(GripperState.FLEXCLOSE)

open_spec_r = right_gripper.default_specification(GripperState.OPEN)
close_spec_r = right_gripper.default_specification(GripperState.CLOSE)
flex_open_spec_r = right_gripper.default_specification(GripperState.FLEXOPEN)
flex_close_spec_r = right_gripper.default_specification(GripperState.FLEXCLOSE)

plan_home = sequential(
    [
        MoveGripperMotion(specification=open_spec_l),
        MoveGripperMotion(specification=open_spec_r),
        parallel(
            [
                MoveGripperMotion(specification=close_spec_l),
                MoveGripperMotion(
                    specification=close_spec_r,
                ),
            ],
        ),
        ParkArmsAction(arm=Arms.RIGHT),
        MoveJointsMotion(
            names=daisy_left_arm_names, positions=daisy_safe_left_arm_positions
        ),
        MoveGripperMotion(specification=flex_close_spec_l),
        MoveGripperMotion(specification=flex_close_spec_r),
        parallel(
            [
                MoveGripperMotion(specification=flex_open_spec_l),
                MoveGripperMotion(specification=flex_open_spec_r),
            ],
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

print("Plan finished.")
