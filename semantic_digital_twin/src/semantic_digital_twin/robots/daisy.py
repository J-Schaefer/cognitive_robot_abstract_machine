from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import ClassVar, List, Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasEndEffector,
    HasLeftRightArm,
    HasTwoFingers,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    EndEffector,
    Finger,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


# ---------------------------------------------------------------------------
# Left gripper fingers
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DaisyLeftGripperLeftFinger(Finger):
    """
    The left finger (thumb) of the left WPG-300 parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """No separate hardware interface for the finger."""

    def setup_joint_states(self) -> List[JointState]:
        """No separate joint states for the finger."""
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_left_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_left_finger_link"
            ),
        )


@dataclass(eq=False)
class DaisyLeftGripperRightFinger(Finger):
    """
    The right finger of the left WPG-300 parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """No separate hardware interface for the finger."""

    def setup_joint_states(self) -> List[JointState]:
        """No separate joint states for the finger."""
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_right_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_right_finger_link"
            ),
        )


# ---------------------------------------------------------------------------
# Right gripper fingers
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DaisyRightGripperLeftFinger(Finger):
    """
    The left finger (thumb) of the right WPG-300 parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """No separate hardware interface for the finger."""

    def setup_joint_states(self) -> List[JointState]:
        """No separate joint states for the finger."""
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_left_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_left_finger_link"
            ),
        )


@dataclass(eq=False)
class DaisyRightGripperRightFinger(Finger):
    """
    The right finger of the right WPG-300 parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """No separate hardware interface for the finger."""

    def setup_joint_states(self) -> List[JointState]:
        """No separate joint states for the finger."""
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_right_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_right_finger_link"
            ),
        )


# ---------------------------------------------------------------------------
# Left gripper
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DaisyLeftGripper(
    EndEffector,
    HasTwoFingers[DaisyLeftGripperLeftFinger, DaisyLeftGripperRightFinger],
):
    """
    The left WPG-300 parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """Sets up hardware interfaces for the gripper's finger joints."""
        for joint_name in (
            "left_gripper_finger_joint",
            "left_gripper_right_finger_joint",
        ):
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """Sets up open and close states for the gripper."""
        gripper_joints = sorted(self.active_connections, key=lambda c: c.name.name)
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.06, -0.06])),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


# ---------------------------------------------------------------------------
# Right gripper
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DaisyRightGripper(
    EndEffector,
    HasTwoFingers[DaisyRightGripperLeftFinger, DaisyRightGripperRightFinger],
):
    """
    The right WPG-300 parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """Sets up hardware interfaces for the gripper's finger joints."""
        for joint_name in (
            "right_gripper_finger_joint",
            "right_gripper_right_finger_joint",
        ):
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """Sets up open and close states for the gripper."""
        gripper_joints = sorted(self.active_connections, key=lambda c: c.name.name)
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.06, -0.06])),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


# ---------------------------------------------------------------------------
# Left arm
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DaisyLeftArm(Arm[DaisyLeftGripper]):
    """
    The left UR5 arm.
    """

    ARM_PARK_CONFIGURATION: ClassVar[dict[str, float]] = {
        "shoulder_pan_joint": -0.26,
        "shoulder_lift_joint": -2.02,
        "elbow_joint": 1.78,
        "wrist_1_joint": -1.28,
        "wrist_2_joint": -1.55,
        "wrist_3_joint": -1.83,
    }

    def setup_hardware_interfaces(self):
        """Sets up hardware interfaces for the arm joints."""
        for joint_name in (
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_elbow_joint",
            "left_wrist_1_joint",
            "left_wrist_2_joint",
            "left_wrist_3_joint",
        ):
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """Sets up the park configuration for the arm."""
        arm_park = JointState.from_mapping(
            name=PrefixedName("park", prefix=self.name.name),
            mapping={
                connection: position
                for connection in self.connections
                if not isinstance(connection, FixedConnection)
                for joint_name, position in self.ARM_PARK_CONFIGURATION.items()
                if connection.name.name.endswith(joint_name)
            },
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_wrist_3_link"
            ),
        )


# ---------------------------------------------------------------------------
# Right arm
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DaisyRightArm(Arm[DaisyRightGripper]):
    """
    The right UR5 arm.
    """

    ARM_PARK_CONFIGURATION: ClassVar[dict[str, float]] = {
        "shoulder_pan_joint": -0.41,
        "shoulder_lift_joint": -1.08,
        "elbow_joint": -1.78,
        "wrist_1_joint": -1.86,
        "wrist_2_joint": 1.57,
        "wrist_3_joint": -1.18,
    }

    def setup_hardware_interfaces(self):
        """Sets up hardware interfaces for the arm joints."""
        for joint_name in (
            "right_shoulder_pan_joint",
            "right_shoulder_lift_joint",
            "right_elbow_joint",
            "right_wrist_1_joint",
            "right_wrist_2_joint",
            "right_wrist_3_joint",
        ):
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """Sets up the park configuration for the arm."""
        arm_park = JointState.from_mapping(
            name=PrefixedName("park", prefix=self.name.name),
            mapping={
                connection: position
                for connection in self.connections
                if not isinstance(connection, FixedConnection)
                for joint_name, position in self.ARM_PARK_CONFIGURATION.items()
                if connection.name.name.endswith(joint_name)
            },
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_wrist_3_link"
            ),
        )


# ---------------------------------------------------------------------------
# DAiSy robot
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DAiSy(AbstractRobot, HasLeftRightArm[DaisyLeftArm, DaisyRightArm]):
    """
    Represents two UR5 arms mounted on a table.
    The left arm is equipped with a WPG-300 parallel gripper.
    The right arm carries a WPG-300 parallel gripper.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        """
        Returns the ROS file path for the DAiSy robot description.
        """
        raise NotImplementedError("We don't have the ROS Package yet")

    @classmethod
    def _get_root_body_name(cls) -> str:
        """
        Returns the name of the root body for the DAiSy robot.
        """
        return "table"

    def _setup_velocity_limits(self):
        """
        Sets up velocity limits for the robot's joints.
        All 1-DOF connections are limited to 0.2.
        """
        vel_limits = defaultdict(lambda: 0.2)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_collision_rules(self):
        """
        Sets up collision avoidance rules for the robot, including SRDF-based
        self-collision ignore rules.
        """
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "daisy.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )
        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.03,
                violated_distance=0.0,
                robot=self,
            )
        )
