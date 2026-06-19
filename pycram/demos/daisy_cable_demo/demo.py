import os

from coraplex.alternative_motion_mappings.daisy_motion_mapping import DAISYGripMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.grasp import GraspDescription
from coraplex.motion_executor import real_robot, simulated_robot
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
import rclpy

from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.testing import setup_world
from coraplex.language import SequentialNode
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, SetGripperAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from test.conftest import world_with_urdf_factory

from define_real_daisy import setup_daisy_context
from define_sim_daisy import setup_sim_daisy

real = True

# %% Robot and World Setup
if real:
    rclpy_node, world, robot_view, context = setup_daisy_context()
else:
    rclpy_node, world, robot_view, context = setup_sim_daisy()

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
bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

with world.modify_world():
    world.merge_world(
        bowl,
        FixedConnection(
            world.root,
            bowl.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
                -0.4, -0.1, 0.635, reference_frame=world.root
            ),
        ),
    )

# %% Visualization
try:
    # rclpy.init()
    # rclpy_node = rclpy.create_node("demo_node")
    v = VizMarkerPublisher(_world=world, node=rclpy_node)
    v.with_tf_publisher()
except ImportError as e:
    print(f"Error: {e}")

# %% Demo
# context = Context.from_world(world)
# daisy = world.get_semantic_annotation_by_name(DAiSy)[0]

print(world.root.name)

# Print joint states
for dof in context.robot.degrees_of_freedom_with_hardware_interface:
    print(f"{dof.name}: {dof.variables.position.resolve()}")

pick_up_grasp = GraspDescription(
    approach_direction=ApproachDirection.FRONT,
    vertical_alignment=VerticalAlignment.TOP,
    end_effector=context.robot.get_left_arm_if_specified().end_effector,
)

plan = sequential(
    [
        ParkArmsAction(arm=Arms.BOTH),
        # PickUpAction(world.get_body_by_name("bowl.stl"), Arms.LEFT, pick_up_grasp),
        # SetGripperAction(gripper=Arms.BOTH, motion=GripperState.CLOSE),
        # SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
        MoveGripperMotion(gripper=Arms.RIGHT, motion=GripperState.CLOSE),
        MoveGripperMotion(gripper=Arms.RIGHT, motion=GripperState.OPEN),
        # Place`Action(
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

while True:
    continue

print("Done.")
