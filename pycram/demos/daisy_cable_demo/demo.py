import os

from pycram.datastructures.dataclasses import Context
from pycram.motion_executor import real_robot, simulated_robot
from pycram.datastructures.enums import Arms
import rclpy

from pycram.plans.factories import sequential
from pycram.plans.plan import Plan
from pycram.testing import setup_world
from pycram.language import SequentialNode
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from test.conftest import world_with_urdf_factory

# %% Environmnent Setup
environment_path = os.path.join("package://iai_apartment/urdf/apartment.urdf")
environment_parser = URDFParser.from_file(file_path=environment_path)
environment_world = environment_parser.parse()

# %% Robot Setup
daisy = "package://iai_daisy_description/robots/daisy.urdf.xacro"
daisy_parser = URDFParser.from_file(file_path=daisy)
daisy_world = daisy_parser.parse()
DAiSy.from_world(daisy_world)

bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

world = daisy_world

world.merge_world(environment_world)

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1, reference_frame=world.root
        ),
    )

# %% Visualization
try:
    import rclpy

    rclpy.init()
    rclpy_node = rclpy.create_node("demo_node")
    v = VizMarkerPublisher(_world=world, node=rclpy_node)
    v.with_tf_publisher()
except ImportError:
    pass

# %% Demo
context = Context.from_world(world)
# daisy = world.get_semantic_annotation_by_name(DAiSy)[0]

print(world.root.name)

with simulated_robot:
    plan = sequential([ParkArmsAction(arm=Arms.BOTH)], context)
    # pose_T_object
    # world.transform(pose_T_object, world.root)
    plan.perform()


while True:
    continue

print("Done.")
