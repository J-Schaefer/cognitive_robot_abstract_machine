import os
import time
from math import pi

from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot, simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.cable_grasp import CableGraspAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

from define_real_daisy import setup_daisy_context
from define_sim_daisy import setup_sim_daisy

real = False

# %% Robot and World Setup
if real:
    node, world, robot_view, context = setup_daisy_context()
else:
    node, world, robot_view, context = setup_sim_daisy()

# %% Define Additional Objects
cup = STLParser(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "resources",
        "objects",
        "jeroen_cup.stl",
    )
).parse()
cup_root = cup.root

with world.modify_world():
    world.merge_world(
        cup,
        FixedConnection(
            world.root,
            cup_root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
                -0.6, -0.1, 0.605, reference_frame=world.root
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
                x=-0.02,
                y=0.025,
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
with world.modify_world():
    cable_annotation = Cable.create_with_new_body_in_world(
        name=PrefixedName("cable"),
        world=world,
        hanging_from=hanger_body,
        length=0.3,
        mount_offset_x=0.078,
        mount_offset_y=-0.05,
        height_offset=0.0,
    )

# %% Demo Plan
pick_up_grasp = GraspDescription(
    approach_direction=ApproachDirection.LEFT,
    vertical_alignment=VerticalAlignment.NoAlignment,
    end_effector=context.robot.get_right_arm_if_specified().end_effector,
    manipulation_offset=0.05,
)

plan = sequential(
    [
        ParkArmsAction(arm=Arms.BOTH),
        SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
        # PickUpAction(
        #     object_designator=cup_root, arm=Arms.RIGHT, grasp_description=pick_up_grasp
        # ),
        CableGraspAction(
            cable_annotation=cable_annotation,
            grasp_offset=0.1,
            approach_offset=0.1,
            approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
        ),
        ParkArmsAction(arm=Arms.BOTH),
    ],
    context,
)

if real:
    with real_robot:
        plan.perform()
else:
    with simulated_robot:
        plan.perform()

print("Plan finished.")
