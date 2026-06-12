import os

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import real_robot, simulated_robot
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
import rclpy

from pycram.plans.factories import sequential
from pycram.plans.plan import Plan
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.testing import setup_world
from pycram.language import SequentialNode
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from pycram.view_manager import ViewManager
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_mixins import SpecifiesLeftRightArm
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from test.conftest import world_with_urdf_factory

from demos.daisy_cable_demo.define_real_daisy import setup_daisy_context
from demos.daisy_cable_demo.define_sim_daisy import setup_sim_daisy

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
    manipulator=context.robot.left_arm.manipulator,
)

plan = sequential(
    [
        ParkArmsAction(arm=Arms.BOTH),
        PickUpAction(world.get_body_by_name("bowl.stl"), Arms.LEFT, pick_up_grasp),
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
