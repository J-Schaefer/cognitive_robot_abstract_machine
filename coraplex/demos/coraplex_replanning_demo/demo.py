#! /usr/bin/env python3

"""
Replanning with an alternative route, at the plan-language level.

The plan parks both arms; a timer ends that motion early, simulating a blocked arm. The
alternative route then returns both arms to the last pose they held before the failure
happened and moves the right arm into the alternative start posture from the
daisy_cable_demo.

Runs DAiSy in a kinematically simulated world and publishes it to Rviz, so the run can
be watched while it happens. Needs the ``iai_daisy_description`` ROS package.

.. note:: The primary route runs as a :class:`~coraplex.language.CodeNode` so the
    ``TryInOrder`` template chart of :class:`~coraplex.language.TryInOrderNode` skips
    it. That chart re-runs after the children ran, and a monitor-wrapped action would
    crash it, because the trigger monitor is re-added there.
"""

# %% imports

import rclpy
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import cancel_when, code, sequential, try_in_order
from coraplex.robot_plans import MoveJointsMotion
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from giskardpy.motion_statechart.monitors.payload_monitors import CountControlCycles
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.daisy import DAiSy

# %% robot setup

daisy_parser = URDFParser.from_file(file_path=DAiSy.get_ros_file_path())
world = daisy_parser.parse()
DAiSy.from_world(world)
daisy = world.get_semantic_annotations_by_type(DAiSy)[0]

rclpy.init()
node = rclpy.create_node("coraplex_replanning_demo")
viz = VizMarkerPublisher(_world=world, node=node)

context = Context(world=world, robot=daisy, ros_node=node)

# %% alternative start posture

right_arm_joint_names = [
    connection.name.name for connection in daisy.right_arm.active_connections
]
safe_right_arm_positions = [
    -0.02,  # right_shoulder_pan_joint
    -0.97,  # right_shoulder_lift_joint
    -2.00,  # right_elbow_joint
    -1.76,  # right_wrist_1_joint
    1.56,  # right_wrist_2_joint
    -0.83,  # right_wrist_3_joint
]

# %% last pose before the failure

pre_failure_joint_names = [
    connection.name.name
    for arm in daisy.all_arms
    for connection in arm.active_connections
]
pre_failure_positions = [
    world.get_connection_by_name(name).position for name in pre_failure_joint_names
]

# %% primary route


def primary_route() -> None:
    """
    Park both arms; a timer ends the motion early, simulating a blocked arm.
    """
    cancel_when(
        [ParkArmsAction(arms=daisy.all_arms)],
        monitor=CountControlCycles(name="failure trigger", control_cycles=50),
        context=context,
    ).perform()


# %% plan

primary = code(primary_route, context)
alternative = sequential(
    [
        MoveJointsMotion(
            names=pre_failure_joint_names, positions=pre_failure_positions
        ),
        MoveJointsMotion(
            names=right_arm_joint_names, positions=safe_right_arm_positions
        ),
    ],
    context,
)
plan = try_in_order([primary, alternative], context).plan

# %% run

with kinematically_simulated_robot():
    plan.perform()

# %% report

print(f"park arms (primary route):  {primary.status.name}")
print(f"alternative route:          {alternative.status.name}")
