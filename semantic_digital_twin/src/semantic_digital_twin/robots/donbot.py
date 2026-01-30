from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from .robot_mixins import HasNeck, HasArms
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    Neck,
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    AbstractRobot,
    JointState,
)
from ..spatial_types import Quaternion, Vector3
from ..world import World
from ..world_description.connections import FixedConnection


@dataclass(eq=False)
class Donbot(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Donbot Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Donbot robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Donbot robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Donbot robot view.
        """

        with world.modify_world():
            donbot = cls(
                name=PrefixedName(name="donbot", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create arm
            gripper_thumb = Finger(
                name=PrefixedName("gripper_thumb", prefix=donbot.name.name),
                root=world.get_body_by_name("gripper_gripper_left_link"),
                tip=world.get_body_by_name("gripper_finger_right_link"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("gripper_finger", prefix=donbot.name.name),
                root=world.get_body_by_name("gripper_gripper_left_link"),
                tip=world.get_body_by_name("gripper_finger_left_link"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=donbot.name.name),
                root=world.get_body_by_name("gripper_base_link"),
                tool_frame=world.get_body_by_name("gripper_tool_frame"),
                front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=gripper_thumb,
                finger=gripper_finger,
                _world=world,
            )
            arm = Arm(
                name=PrefixedName("arm", prefix=donbot.name.name),
                root=world.get_body_by_name("ur5_base_link"),
                tip=world.get_body_by_name("ur5_wrist_3_link"),
                manipulator=gripper,
                _world=world,
            )

            donbot.add_arm(arm)

            # Create camera and neck
            camera = Camera(
                name=PrefixedName("camera_link", prefix=donbot.name.name),
                root=world.get_body_by_name("camera_link"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.5,
                maximal_height=1.2,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=donbot.name.name),
                sensors={camera},
                root=world.get_body_by_name("ur5_base_link"),
                tip=world.get_body_by_name("ur5_base_link"),
                _world=world,
            )
            donbot.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=donbot.name.name),
                root=world.get_body_by_name("base_footprint"),
                tip=world.get_body_by_name("ur5_base_link"),
                _world=world,
            )
            donbot.add_torso(torso)

            # Create states
            arm_park = JointState(
                name=PrefixedName("arm_park", prefix=donbot.name.name),
                joints=[c for c in arm.connections if type(c) != FixedConnection],
                joint_positions=[3.23, -1.51, -1.57, 0.0, 1.57, -1.65],
                state_type=StaticJointState.PARK,
                _world=world,
            )

            arm.add_joint_state(arm_park)

            looking = JointState(
                name=PrefixedName("looking", prefix=donbot.name.name),
                joints=[c for c in neck.connections if type(c) != FixedConnection],
                joint_positions=[0.0, -0.35, -2.15, -0.7, 1.57, -1.57],
                state_type=StaticJointState.PARK,
                _world=world,
            )

            neck.add_joint_state(looking)

            gripper_joints = [
                world.get_connection_by_name("gripper_joint"),
                world.get_connection_by_name("gripper_base_gripper_left_joint"),
            ]

            gripper_open = JointState(
                name=PrefixedName("gripper_open", prefix=donbot.name.name),
                joints=gripper_joints,
                joint_positions=[0.109, -0.055],
                state_type=GripperState.OPEN,
                _world=world,
            )

            gripper_close = JointState(
                name=PrefixedName("gripper_close", prefix=donbot.name.name),
                joints=gripper_joints,
                joint_positions=[0.0065, -0.0027],
                state_type=GripperState.CLOSE,
                _world=world,
            )

            gripper.add_joint_state(gripper_close)
            gripper.add_joint_state(gripper_open)

            torso_joint = [world.get_connection_by_name("arm_base_mounting_joint")]

            torso_low = JointState(
                name=PrefixedName("torso_low", prefix=donbot.name.name),
                joints=torso_joint,
                joint_positions=[0.0],
                state_type=TorsoState.LOW,
                _world=world,
            )

            torso_mid = JointState(
                name=PrefixedName("torso_mid", prefix=donbot.name.name),
                joints=torso_joint,
                joint_positions=[0.0],
                state_type=TorsoState.MID,
                _world=world,
            )

            torso_high = JointState(
                name=PrefixedName("torso_high", prefix=donbot.name.name),
                joints=torso_joint,
                joint_positions=[0.0],
                state_type=TorsoState.HIGH,
                _world=world,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            world.add_semantic_annotation(donbot)

        return donbot
