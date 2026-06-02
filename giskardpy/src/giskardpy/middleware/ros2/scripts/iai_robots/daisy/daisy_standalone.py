import argparse
from threading import Thread

from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    WorldWithDaisyConfig,
    DaisyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.utils.utils import load_xacro
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
)


def main():
    parser = argparse.ArgumentParser(description="DAISY Giskard standalone controller.")
    parser.add_argument(
        "--interactive-marker",
        action="store_true",
        help="Also start the interactive marker server for Cartesian control via RViz.",
    )
    # parse_known_args ignores ROS 2 arguments (--ros-args ...) that argparse does not know about.
    args, _ = parser.parse_known_args()

    rospy.init_node("giskard")
    default_robot_desc = load_xacro(
        "package://iai_daisy_description/robots/daisy.urdf.xacro"
    )
    rospy.node.declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = rospy.node.get_parameter_or("robot_description").value
    if robot_description is None:
        robot_description = default_robot_desc
    giskard = Giskard(
        world_config=WorldWithDaisyConfig(urdf=robot_description),
        robot_interface_config=DaisyStandAloneRobotInterfaceConfig(),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )

    if args.interactive_marker:
        Thread(
            target=lambda: InteractiveMarkerNode(
                root_links=["map", "map"],
                tip_links=[
                    "left_gripper_tool_frame",
                    "right_gripper_tool_frame",
                ],
            ),
            daemon=True,
            name="interactive_marker",
        ).start()

    giskard.live()


if __name__ == "__main__":
    main()
