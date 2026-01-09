from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rclpy.node import MsgType
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from typing_extensions import Generic, Type

import krrood.symbolic_math.symbolic_math as sm
from ..context import ExecutionContext, BuildContext
from ..data_types import ObservationStateValues
from ..graph_node import MotionStatechartNode, NodeArtifacts
from ..ros_context import RosContextExtension


@dataclass
class TopicSubscriberNode(MotionStatechartNode, Generic[MsgType]):
    """
    Superclass for all nodes that subscribe to a ROS topic.
    This node will automatically create a subscriber on build and cache the last message in `current_msg` on_tick.
    If you overwrite `on_tick`, make sure to call `super().on_tick(context)` first.
    """

    topic_name: str = field(kw_only=True)
    """Name of the ROS topic to subscribe to."""
    msg_type: Type[MsgType] = field(kw_only=True)
    """Type of the ROS message."""
    qos_profile: QoSProfile | int = field(kw_only=True, default=10)
    """QoS profile to use when subscribing to the topic."""
    _subscriber: Subscription = field(init=False)
    """Internal ROS subscription object."""
    __last_msg: MsgType | None = field(init=False, default=None)
    """
    The callback updates this variable.
    Don't use it directly, use `current_msg` instead.
    """
    current_msg: MsgType | None = field(init=False, default=None)
    """
    __last_msg is copied to this variable on every tick while this node is RUNNING.
    """

    def build(self, context: BuildContext) -> NodeArtifacts:
        ros_context_extension = context.require_extension(RosContextExtension)
        self._subscriber = ros_context_extension.ros_node.create_subscription(
            msg_type=self.msg_type,
            topic=self.topic_name,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )
        return NodeArtifacts()

    def callback(self, msg: MsgType):
        self.__last_msg = msg

    def has_msg(self) -> bool:
        return self.current_msg is not None

    def clear_msg(self):
        self.__last_msg = None

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        self.current_msg = self.__last_msg

    def on_reset(self, context: ExecutionContext):
        self.clear_msg()


@dataclass(eq=False, repr=False)
class TopicPublisherNode(MotionStatechartNode, Generic[MsgType]):
    """
    Superclass for all nodes that publish to a ROS topic.
    This node will automatically create a publisher on build.
    """
    topic_name: str = field(kw_only=True)
    """Name of the ROS topic to publish to."""
    msg_type: Type[MsgType] = field(kw_only=True)
    """Type of the ROS message."""
    qos_profile: QoSProfile | int = field(kw_only=True, default=10)
    """QoS profile to use when publishing to the topic."""
    _publisher: Publisher = field(init=False)
    """Internal ROS publisher object."""

    def build(self, context: BuildContext) -> NodeArtifacts:
        ros_context_extension = context.require_extension(RosContextExtension)
        self._publisher = ros_context_extension.ros_node.create_publisher(
            msg_type=self.msg_type,
            topic=self.topic_name,
            qos_profile=self.qos_profile,
        )
        return NodeArtifacts()


@dataclass(eq=False, repr=False)
class WaitForMessage(TopicSubscriberNode[MsgType]):
    """
    This node will turn to True once a message was received on its topic.
    """

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        super().on_tick(context)
        if self.has_msg():
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False)
class PublishOnStart(TopicPublisherNode[MsgType]):
    """
    This node will publish its message when on_start is called.
    This is not repeated on every tick, but will be repeated after a reset, if the node is started again.
    """
    msg: MsgType = field(kw_only=True)
    """Message to publish."""
    msg_type: Type[MsgType] = field(init=False)
    """init=False, because we can figure out the type from the msg parameter."""

    def __post_init__(self):
        super().__post_init__()
        self.msg_type = type(self.msg)

    def build(self, context: BuildContext) -> NodeArtifacts:
        node_artifacts = super().build(context)
        node_artifacts.observation = sm.Scalar.const_true()
        return node_artifacts

    def on_start(self, context: ExecutionContext):
        self._publisher.publish(self.msg)
