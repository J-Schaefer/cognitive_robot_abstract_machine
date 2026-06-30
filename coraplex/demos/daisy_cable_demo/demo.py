import os
from math import pi

from coraplex.alternative_motion_mappings.daisy_motion_mapping import DAISYGripMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.grasp import GraspDescription
from coraplex.motion_executor import real_robot, simulated_robot
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
import rclpy

from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveGripperMotion
import coraplex.alternative_motion_mappings.daisy_motion_mapping  # type: ignore
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.testing import setup_world
from coraplex.language import SequentialNode
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from test.conftest import world_with_urdf_factory

from define_real_daisy import setup_daisy_context
from define_sim_daisy import setup_sim_daisy

# %% Environmental Variables Setup

real = True

# %% Robot and World Setup
if real:
    node, world, robot_view, context = setup_daisy_context()
else:
    node, world, robot_view, context = setup_sim_daisy()

# %% Environment Setup
# if not real:
#     apartment_path = os.path.join("package://iai_apartment/urdf/apartment.urdf")
#     apartment_parser = URDFParser.from_file(file_path=apartment_path)
#     apartment_world = apartment_parser.parse()
#
#     with world.modify_world():
#         world.merge_world_at_pose(
#             apartment_world,
#             HomogeneousTransformationMatrix.from_xyz_rpy(
#                 2, -5, 0, 0, 0, 3.14 / 2, reference_frame=world.root
#             ),
#         )

# %% Define Additional Objects
try:

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

except Exception as e:
    print(e)
    print(
        "Bowl already exists in the world. Using existing bowl instead of creating a new one."
    )
    bowl = world.get_body_by_name(PrefixedName("bowl"))


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
cable_post_root = (
    cable_post.root
)  # get object root because it is cleared and becomes None after merging

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
cable_hanger_root = (
    cable_hanger.root
)  # get object root because it is cleared and becomes None after merging

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
                # yaw=math.pi,
                reference_frame=cable_post_root,
            ),
        ),
    )

# %% Demo
# context = Context.from_world(world)
# daisy = world.get_semantic_annotation_by_name(DAiSy)[0]

print(world.root.name)
print([body.name for body in world.bodies])

# Print joint states
for dof in context.robot.degrees_of_freedom_with_hardware_interface:
    print(f"{dof.name}: {dof.variables.position.resolve()}")

pick_up_grasp = GraspDescription(
    approach_direction=ApproachDirection.FRONT,
    vertical_alignment=VerticalAlignment.TOP,
    end_effector=context.robot.get_left_arm_if_specified().end_effector,
    manipulation_offset=0.05,
)
# pick_up_grasp.grasp_pose()  # Body for geometry
# pick_up_grasp._pose_sequence()

plan = sequential(
    [
        ParkArmsAction(arm=Arms.BOTH),
        # SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
        # SetGripperAction(gripper=Arms.BOTH, motion=GripperState.CLOSE),
        # SetGripperAction(gripper=Arms.RIGHT, motion=GripperState.CLOSE),
        SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
        PickUpAction(world.get_body_by_name("jeroen_cup.stl"), Arms.LEFT, pick_up_grasp),
        # PlaceAction(
        #     world.get_body_by_name("bowl.stl"),
        #     HomogeneousTransformationMatrix.from_xyz_rpy(
        #         -0.6, -0.1, 0.635, reference_frame=world.root
        #     ).to_pose(),
        #     Arms.LEFT,
        # ),
    ],
    context,
)
# pose_T_object
# world.transform(pose_T_object, world.root)

if real:
    with real_robot:
        plan.perform()
else:
    with simulated_robot:
        plan.perform()

print("Plan finished.")

# while True:
#     continue

print("Done.")
