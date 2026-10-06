from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.robots.gripper_configurations import (
    GriplinkGripperConfiguration,
)
from semantic_digital_twin.datastructures.robots.gripper_specification import (
    MAXIMUM_OPENING_WIDTH_MM,
    GriplinkFlexSpecification,
    GripperStateSpecification,
)
from semantic_digital_twin.exceptions import (
    ConnectionsOutsideEndEffector,
    MissingPositionLimits,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)

# %% GripperStateSpecification


def test_closed_specification_carries_the_end_effectors_own_joint_state(daisy_world):
    """
    ``GripperStateSpecification.closed(gripper).joint_state`` is the very object
    ``gripper.get_joint_state_by_type(GripperState.CLOSE)`` returns -- not a copy.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.get_end_effectors():
        expected = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        specification = GripperStateSpecification.closed(end_effector)
        assert specification.joint_state is expected


def test_specification_rejects_foreign_connections(daisy_world):
    """
    A specification built with a joint state from the other arm's gripper raises
    ``ConnectionsOutsideEndEffector``.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    left_gripper = daisy.left_arm.end_effector
    right_gripper = daisy.right_arm.end_effector

    right_close_state = right_gripper.get_joint_state_by_type(GripperState.CLOSE)
    with pytest.raises(ConnectionsOutsideEndEffector):
        GripperStateSpecification(
            end_effector=left_gripper,
            joint_state=right_close_state,
        )


def test_specification_end_effector_and_joint_state_are_carried(daisy_world):
    """
    The specification carries the end effector and the joint state it was built from.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)

    specification = GripperStateSpecification.closed(end_effector)
    assert specification.end_effector is end_effector
    assert specification.joint_state is close_state


# %% mimics for the griplink flex specification


@dataclass
class DegreeOfFreedomWithoutPositionLimits:
    """
    Mimics a degree of freedom whose limits declare no positions, the only part of a
    degree of freedom the flex joint state builder reads.
    """

    name: PrefixedName = field(
        default_factory=lambda: PrefixedName(
            "degree_of_freedom_without_position_limits"
        )
    )
    """
    The name of the degree of freedom.
    """

    limits: DegreeOfFreedomLimits = field(default_factory=DegreeOfFreedomLimits)
    """
    The limits of the degree of freedom, without positions.
    """


@dataclass
class ConnectionWithoutPositionLimits:
    """
    Mimics a connection whose degree of freedom declares no position limits, the only
    part of a connection the flex joint state builder reads.
    """

    dof: DegreeOfFreedomWithoutPositionLimits = field(
        default_factory=DegreeOfFreedomWithoutPositionLimits
    )
    """
    The degree of freedom of the connection.
    """


class GripperWithoutPositionLimits:
    """
    Mimics a detached gripper whose open state holds a connection without position
    limits; detached so the specification skips its end effector validation.
    """

    _world = None
    """
    Detached from a world, so the specification skips its end effector validation.
    """

    def __init__(self):
        self.name = PrefixedName("gripper_without_position_limits")
        """
        The name of the gripper.
        """

        self._open_state = JointState(
            connections=[ConnectionWithoutPositionLimits()],
            state_type=GripperState.OPEN,
        )

    def get_joint_state_by_type(self, state_type: GripperState) -> JointState:
        """
        Returns the open joint state holding the connection without position limits.
        """
        return self._open_state


# %% GriplinkFlexSpecification


def test_flex_specification_interpolates_the_open_state_by_the_grip_position(
    daisy_world,
):
    """
    A flex specification commands, for every connection of the open state, the point
    between its position limits that the configured opening width corresponds to.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)

    configuration = GriplinkGripperConfiguration(grip_position=60)
    specification = GriplinkFlexSpecification.from_state_type(
        end_effector, GripperState.FLEXCLOSE, configuration
    )

    fraction = (MAXIMUM_OPENING_WIDTH_MM - 60) / MAXIMUM_OPENING_WIDTH_MM
    expected = [
        connection.dof.limits.lower.position
        + fraction
        * (connection.dof.limits.upper.position - connection.dof.limits.lower.position)
        for connection in open_state.connections
    ]
    assert specification.joint_state.target_values == pytest.approx(expected)


def test_flex_specification_without_grip_position_commands_the_open_state(daisy_world):
    """
    Without a configured opening width, a flex specification commands the lower
    position limits, i.e. the fully opened state.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)

    specification = GriplinkFlexSpecification.from_state_type(
        end_effector, GripperState.FLEXOPEN
    )

    expected = [
        connection.dof.limits.lower.position for connection in open_state.connections
    ]
    assert specification.joint_state.target_values == pytest.approx(expected)


def test_flexclose_specification_without_grip_position_commands_the_fully_closed_state(
    daisy_world,
):
    """
    Without a configured opening width, a ``FLEXCLOSE`` specification commands the
    upper position limits, i.e. the fully closed state.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)

    specification = GriplinkFlexSpecification.from_state_type(
        end_effector, GripperState.FLEXCLOSE
    )

    expected = [
        connection.dof.limits.upper.position for connection in open_state.connections
    ]
    assert specification.joint_state.target_values == pytest.approx(expected)


def test_flex_specification_rejects_connections_without_position_limits():
    """
    Interpolating a connection whose degree of freedom declares no position limits
    cannot yield a meaningful target and raises ``MissingPositionLimits``.
    """
    gripper = GripperWithoutPositionLimits()

    with pytest.raises(MissingPositionLimits):
        GriplinkFlexSpecification.from_state_type(gripper, GripperState.FLEXCLOSE)
