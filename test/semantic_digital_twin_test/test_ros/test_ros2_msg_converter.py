import pytest
from geometry_msgs.msg import (
    TransformStamped,
    PointStamped,
    QuaternionStamped,
    Vector3Stamped,
    PoseStamped,
)

from semantic_digital_twin.adapters.ros.msg_converter import (
    ROS2MessageConverter,
    CannotConvertFromRos2Error,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Box, Color, Scale


def test_convert_transform(cylinder_bot_world):
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "bot"
    transform.transform.translation.x = 1.0
    transform.transform.rotation.x = 1.0
    transform.transform.rotation.w = 0.0
    transformation_matrix = ROS2MessageConverter.from_ros2_message(
        transform, world=cylinder_bot_world
    )
    position = transformation_matrix.to_position().evaluate()
    rotation = transformation_matrix.to_quaternion().evaluate()
    assert position[0] == transform.transform.translation.x
    assert position[1] == transform.transform.translation.y
    assert position[2] == transform.transform.translation.z

    assert rotation[0] == transform.transform.rotation.x
    assert rotation[1] == transform.transform.rotation.y
    assert rotation[2] == transform.transform.rotation.z
    assert rotation[3] == transform.transform.rotation.w
    assert transformation_matrix.child_frame == cylinder_bot_world.get_body_by_name(
        "bot"
    )
    assert transformation_matrix.reference_frame == cylinder_bot_world.get_body_by_name(
        "map"
    )

    transform2 = ROS2MessageConverter.to_ros2_message(transformation_matrix)
    assert transform == transform2


def test_convert_point_stamped(cylinder_bot_world):
    point_msg = PointStamped()
    point_msg.header.frame_id = "map"
    point_msg.point.x = 1.0
    point_msg.point.y = 2.0
    point_msg.point.z = 3.0

    point = ROS2MessageConverter.from_ros2_message(point_msg, world=cylinder_bot_world)
    coords = point.evaluate()

    assert coords[0] == point_msg.point.x
    assert coords[1] == point_msg.point.y
    assert coords[2] == point_msg.point.z

    assert point.reference_frame == cylinder_bot_world.get_body_by_name("map")

    point_msg2 = ROS2MessageConverter.to_ros2_message(point)
    assert point_msg == point_msg2


def test_convert_quaternion(cylinder_bot_world):
    quat_msg = QuaternionStamped()
    quat_msg.header.frame_id = "map"
    quat_msg.quaternion.x = 1.0
    quat_msg.quaternion.y = 0.0
    quat_msg.quaternion.z = 0.0
    quat_msg.quaternion.w = 0.0

    quat = ROS2MessageConverter.from_ros2_message(quat_msg, world=cylinder_bot_world)
    values = quat.evaluate()

    assert values[0] == quat_msg.quaternion.x
    assert values[1] == quat_msg.quaternion.y
    assert values[2] == quat_msg.quaternion.z
    assert values[3] == quat_msg.quaternion.w

    assert quat.reference_frame == cylinder_bot_world.get_body_by_name("map")

    quat_msg2 = ROS2MessageConverter.to_ros2_message(quat)
    assert quat_msg == quat_msg2


def test_convert_vector3(cylinder_bot_world):
    vec_msg = Vector3Stamped()
    vec_msg.header.frame_id = "map"
    vec_msg.vector.x = -1.0
    vec_msg.vector.y = 0.5
    vec_msg.vector.z = 2.5

    vec = ROS2MessageConverter.from_ros2_message(vec_msg, world=cylinder_bot_world)
    values = vec.evaluate()

    assert values[0] == vec_msg.vector.x
    assert values[1] == vec_msg.vector.y
    assert values[2] == vec_msg.vector.z

    assert vec.reference_frame == cylinder_bot_world.get_body_by_name("map")

    vec_msg2 = ROS2MessageConverter.to_ros2_message(vec)
    assert vec_msg == vec_msg2


def test_convert_pose_stamped(cylinder_bot_world):
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "map"
    pose_msg.pose.position.x = 1.2
    pose_msg.pose.position.y = -0.4
    pose_msg.pose.position.z = 0.7
    pose_msg.pose.orientation.x = 0.0
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 1.0

    pose = ROS2MessageConverter.from_ros2_message(pose_msg, world=cylinder_bot_world)

    pos = pose.to_position().evaluate()
    quat = pose.to_quaternion().evaluate()

    assert pos[0] == pose_msg.pose.position.x
    assert pos[1] == pose_msg.pose.position.y
    assert pos[2] == pose_msg.pose.position.z

    assert quat[0] == pose_msg.pose.orientation.x
    assert quat[1] == pose_msg.pose.orientation.y
    assert quat[2] == pose_msg.pose.orientation.z
    assert quat[3] == pose_msg.pose.orientation.w

    assert pose.reference_frame == cylinder_bot_world.get_body_by_name("map")

    pose_msg2 = ROS2MessageConverter.to_ros2_message(pose)
    assert pose_msg == pose_msg2


def test_no_frame_id(cylinder_bot_world):
    ROS2MessageConverter.to_ros2_message(HomogeneousTransformationMatrix())
    ROS2MessageConverter.to_ros2_message(Pose())
    ROS2MessageConverter.to_ros2_message(Point3())
    ROS2MessageConverter.to_ros2_message(Vector3())
    ROS2MessageConverter.to_ros2_message(Quaternion())

    ROS2MessageConverter.from_ros2_message(TransformStamped(), cylinder_bot_world)
    ROS2MessageConverter.from_ros2_message(PointStamped(), cylinder_bot_world)
    ROS2MessageConverter.from_ros2_message(QuaternionStamped(), cylinder_bot_world)
    ROS2MessageConverter.from_ros2_message(Vector3Stamped(), cylinder_bot_world)
    ROS2MessageConverter.from_ros2_message(PoseStamped(), cylinder_bot_world)


def test_convert_shape(cylinder_bot_world):
    box = Box(
        scale=Scale(1, 1, 1),
        color=Color(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 2, 3, 1, 2, 3, reference_frame=cylinder_bot_world.get_body_by_name("map")
        ),
    )
    shape = ROS2MessageConverter.to_ros2_message(box)

    # with pytest.raises(CannotConvertFromRos2Error):
    box2 = ROS2MessageConverter.from_ros2_message(shape, cylinder_bot_world)
