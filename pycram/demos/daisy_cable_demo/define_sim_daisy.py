from typing import Any

from coraplex.datastructures.dataclasses import Context
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.daisy import DAiSy

import rclpy

from semantic_digital_twin.world import World


def setup_sim_daisy(
    node_name: str = "coraplex_node",
) -> tuple[Any, World, DAiSy, Context]:

    rospy.init_node("demo_node")


    # %% Robot Setup
    daisy = "package://iai_daisy_description/robots/daisy.urdf.xacro"
    daisy_parser = URDFParser.from_file(file_path=daisy)
    daisy_world = daisy_parser.parse()
    DAiSy.from_world(daisy_world)

    world = daisy_world

    viz = VizMarkerPublisher(_world=world, node=rospy.node)
    viz.with_tf_publisher()

    # Robot semantic view
    robot_view = world.get_semantic_annotations_by_type(DAiSy)[0]

    # Context
    context = Context(
        world,
        robot_view,
    )

    return rospy.node, world, robot_view, context
