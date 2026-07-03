import logging
import os
import time
from math import pi

from coraplex.alternative_motion_mappings.daisy_motion_mapping import DAISYGripMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.motion_executor import real_robot, simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.cable import CableConfig, CableSimulation
from semantic_digital_twin.world_description.connections import FixedConnection

from define_real_daisy import setup_daisy_context
from define_sim_daisy import setup_sim_daisy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

real = False

# %% Robot and World Setup
if real:
    node, world, robot_view, context = setup_daisy_context()
else:
    node, world, robot_view, context = setup_sim_daisy()

# %% Define Additional Objects
cup = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "jeroen_cup.stl"
    )
).parse()
cup_root = cup.root

with world.modify_world():
    world.merge_world(
        cup,
        FixedConnection(
            world.root,
            cup.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
                -0.6, -0.1, 0.68, reference_frame=world.root
            ),
        ),
    )

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
                x=0.02,
                y=-0.02,
                z=0.8,
                roll=pi / 2,
                reference_frame=world.get_semantic_annotations_by_type(DAiSy)[0].root,
            ),
        ),
    )

cable_hanger = STLParser(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "resources",
        "objects",
        "cable_hanger_item.stl",
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
                x=-0.02,
                y=0.3,
                z=0.0,
                roll=-pi / 2,
                pitch=-pi / 2,
                reference_frame=cable_post_root,
            ),
        ),
    )

# %% Build cable and start background physics simulation
cable_config = CableConfig(
    segment_count=12,
    segment_length=0.04,
    radius=0.006,
    mass_per_segment=0.005,
)
cable_sim = CableSimulation(config=cable_config, world=world)
cable_sim.start()

# %% Demo Plan
pick_up_grasp = GraspDescription(
    approach_direction=ApproachDirection.BACK,
    vertical_alignment=VerticalAlignment.NoAlignment,
    end_effector=context.robot.get_right_arm_if_specified().end_effector,
    manipulation_offset=0.05,
)

plan = sequential(
    [
        ParkArmsAction(arm=Arms.BOTH),
        SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
    ],
    context,
)

if real:
    with real_robot:
        plan.perform()
else:
    with simulated_robot:
        plan.perform()

print("Plan finished. Background cable simulation is running.")

# %% Demonstrate cable grasp and release in the simulation
time.sleep(1.0)
try:
    cable_sim.grasp(gripper_body_name="right_gripper_tool_frame", segment_index=0)
    print("Cable segment 0 grasped by right gripper tool frame.")
    time.sleep(2.0)
    cable_positions = cable_sim.get_segment_positions()
    print(f"Cable segment 0 position after grasp: {cable_positions['cable_segment_0']}")
except Exception as e:
    logger.warning("Cable grasp demonstration failed: %s", e)

print("Done.")
