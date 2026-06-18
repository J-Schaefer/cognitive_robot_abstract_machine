from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from geometry_msgs.msg import (
    PoseStamped as ROSPoseStamped,
    Pose as ROSPose,
    Point as ROSPoint,
    Quaternion as ROSQuaternion,
)

from griplink_interfaces.action import Grip, Release, Flexgrip, Flexrelease

from coraplex.datastructures.enums import WPGGripPreset
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import EndEffector

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None
from rclpy.action import ActionClient
from std_msgs.msg import Header
from typing_extensions import Type, TypeVar, Generic

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.ros_context import RosContextExtension
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


Action = TypeVar("Action")
ActionGoal = TypeVar("ActionGoal")
ActionResult = TypeVar("ActionResult")
ActionFeedback = TypeVar("ActionFeedback")


@dataclass(eq=False, repr=False)
class ActionServerTask(
    MotionStatechartNode,
    ABC,
    Generic[Action, ActionGoal, ActionResult, ActionFeedback],
):
    """
    Abstract base class for tasks that call a ROS2 action server.
    """

    action_topic: str
    """
    Topic name for the action server.
    """

    message_type: Type[Action]
    """
    Fully specified goal message that can be send out.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS action client, is created in `build`.
    """

    _msg: ActionGoal = field(init=False, default=None)
    """
    ROS message to send to the action server.
    """

    _result: ActionResult = field(init=False, default=None)
    """
    ROS action server result.
    """

    @abstractmethod
    def build_msg(self, context: MotionStatechartContext):
        """
        Build the action server message and returns it.
        """
        ...

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the action client.
        """
        ros_context_extension = context.require_extension(RosContextExtension)
        self._action_client = ros_context_extension.get_or_create_action_client(
            self.message_type, self.action_topic
        )
        self.build_msg(context)
        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()
        return super().build(context)

    def on_start(self, context: MotionStatechartContext):
        """
        Creates a goal and sends it to the action server asynchronously.
        """
        future = self._action_client.send_goal_async(self._msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handles the server's response to the goal submission.

        On rejection a failure sentinel is stored so that :meth:`on_tick` can return
        :attr:`~ObservationStateValues.FALSE` immediately.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Goal rejected by action server")
            return

        logger.info("Sent query to action server ")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self._result = future.result()
        logger.info(f"Action server {self.action_topic} done.")


@dataclass(eq=False, repr=False)
class NavigateActionServerTask(
    ActionServerTask[
        NavigateToPose,
        NavigateToPose.Goal,
        NavigateToPose.Result,
        NavigateToPose.Feedback,
    ]
):
    """
    Node for calling a Navigation2 ROS2 action server to navigate to a given pose.
    """

    target_pose: Pose
    """
    Target pose to which the robot should navigate.
    """

    base_link: Body
    """
    Base link of the robot, used for estimating the distance to the goal.
    """

    def build_msg(self, context: MotionStatechartContext):
        root_p_goal = context.world.transform(
            target_frame=context.world.root, spatial_object=self.target_pose
        )
        position = root_p_goal.to_position().to_np()
        orientation = root_p_goal.to_quaternion().to_np()
        pose_stamped = ROSPoseStamped(
            header=Header(frame_id="map"),
            pose=ROSPose(
                position=ROSPoint(x=position[0], y=position[1], z=position[2]),
                orientation=ROSQuaternion(
                    x=orientation[0],
                    y=orientation[1],
                    z=orientation[2],
                    w=orientation[3],
                ),
            ),
        )
        self._msg = NavigateToPose.Goal(pose=pose_stamped)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Observes whether the robot is within 1cm of the target pose.
        """
        artifacts = NodeArtifacts()
        root_T_goal = context.world.transform(
            target_frame=context.world.root, spatial_object=self.target_pose
        )
        root_T_current = context.world.compose_forward_kinematics_expression(
            context.world.root, self.base_link
        )

        position_error = root_T_goal.to_position().euclidean_distance(
            root_T_current.to_position()
        )
        rotation_error = root_T_goal.to_rotation_matrix().rotational_error(
            root_T_current.to_rotation_matrix()
        )

        artifacts.observation = sm.trinary_logic_and(
            position_error < 0.01, sm.abs(rotation_error) < 0.01
        )

        return artifacts

    def result_callback(self, future):
        """
        Stores the navigation result returned by the action server.
        """
        # TODO: Check the ._result.result calls against a running action server because I'm not sure if all of them are correct
        super().result_callback(future)
        logger.info(
            f"Finished navigation with response status: {self._result.result.status} and result code: {self._result.error_code}"
        )

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        if self._result.result:
            return (
                ObservationStateValues.TRUE
                if self._result.error_code == NavigateToPose.Result.NONE
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN


class WPGGripperActionServerTask(
    ActionServerTask[Grip, Grip.Goal, Grip.Result, Grip.Feedback]
):
    """
    Node for calling a WPG-300 ROS2 action server to grip the object.
    """

    grip_preset: WPGGripPreset = WPGGripPreset.PRESET_0
    """
    Grip preset.
    """

    grip_position: int = None
    """
    Opening width of the gripper [-5..120 mm].
    """

    grip_force: int = None
    """
    Force the gripper applies to the object [30..300 N].
    """

    grip_speed: int = None
    """
    Motion speed of the gripper [5..350 mm/s].
    """

    grip_acceleration: int = None
    """
    Motion acceleration of the gripper [100..4000 mm/s^2].
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Creates and returns a message based on the provided MotionStatechartContext.

        The method processes the given context to construct a specific message
        that can be utilized for further communication or logging purposes. The
        context determines the message's content and structure.

        Parameters:
            context: MotionStatechartContext
                The context from which the message is built. It contains information
                necessary to construct the message.

        Returns:
            str: The constructed message based on the provided context.
        """
        super().build_msg(context)

        if self.message_type == Flexgrip:
            self._msg = Flexgrip.Goal(
                port=0,
                position=self.grip_position,
                force=self.grip_force,
                speed=self.grip_speed,
                acceleration=self.grip_acceleration,
            )
        elif self.message_type == Flexrelease:
            self._msg = Flexrelease.Goal(
                port=0,
                position=self.grip_position,
                speed=self.grip_speed,
                acceleration=self.grip_acceleration,
            )
        elif self.message_type == Grip:
            self._msg = Grip.Goal(
                port=0,
                index=self.grip_preset,
            )
        elif self.message_type == Release:
            self._msg = Release.Goal(
                port=0,
                index=self.grip_preset,
            )
        else:
            raise ValueError(f"Unknown message type: {self.message_type}")

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Builds the required NodeArtifacts for the given context.

        This method is used to construct the necessary node artifacts based on
        the provided motion statechart context. It overrides the parent class
        method to include specific message building functionality.

        Parameters:
        context (MotionStatechartContext): The motion statechart context used
        to build the artifacts. Must be an instance of MotionStatechartContext.

        Returns:
        NodeArtifacts: The constructed node artifacts.
        """
        super().build_msg(context)

        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        if self._result:
            if self.message_type == Flexgrip:
                return (
                    ObservationStateValues.TRUE
                    if self._result.error_code == Flexgrip.Result.NONE
                    else ObservationStateValues.FALSE
                )
            elif self.message_type == Flexrelease:
                return (
                    ObservationStateValues.TRUE
                    if self._result.error_code == Flexrelease.Result.NONE
                    else ObservationStateValues.FALSE
                )
            elif self.message_type == Grip:
                return (
                    ObservationStateValues.TRUE
                    if self._result.error_code == Grip.Result.NONE
                    else ObservationStateValues.FALSE
                )
            elif self.message_type == Release:
                return (
                    ObservationStateValues.TRUE
                    if self._result.error_code == Release.Result.NONE
                    else ObservationStateValues.FALSE
                )
            else:
                raise ValueError(f"Unknown message type: {self.message_type}")
        return ObservationStateValues.UNKNOWN
