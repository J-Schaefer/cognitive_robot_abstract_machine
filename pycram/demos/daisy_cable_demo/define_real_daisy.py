import threading
import time
from dataclasses import dataclass

import logging

from typing_extensions import Tuple, Any

from coraplex.alternative_motion_mappings.daisy_motion_mapping import DAiSyGripMotion
from coraplex.datastructures.dataclasses import Context
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    WorldSynchronizer,
    ModelReloadSynchronizer,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)


def setup_real_daisy(
    node_name: str = "coraplex_node",
) -> Tuple[Any, World, DAiSy, Context]:
    """
    Initializes rclpy, starts a SingleThreadedExecutor in a background thread,
    synchronizes the world model, and returns all relevant objects.

    Returns:
        dict containing:
            - node
            - world
            - robot_view
            - context
    """
    rospy.init_node("demo_node")

    # Fetch world
    world: World = fetch_world_from_service(rospy.node)

    # Synchronizer
    world_sync = WorldSynchronizer(_world=world, node=rospy.node)
    ModelReloadSynchronizer(_world=world, node=rospy.node)

    # Optional TF publisher
    # TFPublisher(world=world, node=rospy.node)

    # env_world = load_environment()
    # with world.modify_world():
    #     world.merge_world(env_world)

    # Visualization
    # VizMarkerPublisher(world=world, node=rospy.node)

    # Robot semantic view
    robot_view = world.get_semantic_annotations_by_type(DAiSy)[0]

    # Context
    context = Context(
        world,
        robot_view,
        ros_node=rospy.node,
        alternative_motion_mappings=[DAiSyGripMotion],
    )

    return rospy.node, world, robot_view, context
