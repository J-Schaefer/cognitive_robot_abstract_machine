from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from typing_extensions import Self, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ConnectionsOutsideEndEffector
from semantic_digital_twin.datastructures.robots.gripper_configurations import (
    GriplinkGripperConfiguration,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import EndEffector


TGripperSpecification = TypeVar("TGripperSpecification", bound="GripperSpecification")
"""
The specification type a motion or mapping is bound to; concrete consumers bind it to
one specification subclass.
"""

# %% Base specification


@dataclass(eq=False)
class GripperSpecification(Generic[TGripperSpecification], SubClassSafeGeneric, ABC):
    """
    A configuration a gripper can be commanded into.

    Carries the end effector the specification configures and the joint positions to
    reach, so both the action and the motion that consume it share one object instead of
    resolving the end effector and looking up the joint state independently.
    """

    end_effector: EndEffector
    """
    The end effector this specification configures.
    """

    joint_state: JointState
    """
    The positions the end effector's connections are commanded to.
    """

    finger_velocity: float | None = field(default=None)
    """
    Maximum finger joint velocity (in m/s) enforced during the motion.

    ``None`` leaves the speed unconstrained.
    """

    @classmethod
    def from_state_type(
        cls, end_effector: EndEffector, state_type: GripperState
    ) -> Self:
        """
        :param end_effector: The end effector whose declared state is used.
        :param state_type: The state type to build the specification for.
        :return: The specification for the declared state of the given type.
        :raises NoJointStateWithType: If the end effector declares no such state.
        """
        return cls(
            end_effector=end_effector,
            joint_state=end_effector.get_joint_state_by_type(state_type),
        )

    def __post_init__(self):
        """
        Validates that the joint state only commands connections inside the end
        effector.

        :return: None
        :raises ConnectionsOutsideEndEffector: If the joint state commands a connection
            outside the end effector.
        """
        if self.end_effector._world is None:
            # The end effector is being reconstructed detached from a world (e.g. from
            # the database), so its connections cannot be looked up yet.
            return
        connections_outside_end_effector = set(self.joint_state.connections) - set(
            self.end_effector.active_connections
        )
        if connections_outside_end_effector:
            raise ConnectionsOutsideEndEffector(
                end_effector=self.end_effector,
                foreign_connection_names=[
                    str(c.name) for c in connections_outside_end_effector
                ],
            )


# %% Default specification


@dataclass(eq=False)
class GripperStateSpecification(GripperSpecification):
    """
    The configuration a robot description already declares for a gripper, selected by
    its :class:`~semantic_digital_twin.datastructures.definitions.GripperState`.
    """

    @classmethod
    def closed(cls, end_effector: EndEffector) -> Self:
        """
        :param end_effector: The end effector to build the specification for.
        :return: The specification for the end effector's closed state.
        """
        return cls.from_state_type(end_effector, GripperState.CLOSE)


# %% Griplink specifications

MAXIMUM_OPENING_WIDTH_MM = 120
"""
Maximum opening width of the griplink gripper in millimetres, the scale the flex
specification interpolates ``grip_position`` on.
"""

FULLY_CLOSED_OPENING_WIDTH_MM = 0
"""
Opening width of the griplink gripper in millimetres that commands the fully closed
state.
"""


@dataclass(eq=False)
class GriplinkSpecification(GripperSpecification, ABC):
    """
    A specification for a griplink gripper, carrying the hardware parameters the
    griplink action server executes the motion with.
    """

    configuration: GriplinkGripperConfiguration = field(
        default_factory=GriplinkGripperConfiguration
    )
    """
    Hardware parameters forwarded to the griplink action server.
    """

    @classmethod
    def from_state_type(
        cls,
        end_effector: EndEffector,
        state_type: GripperState,
        configuration: GriplinkGripperConfiguration | None = None,
    ) -> Self:
        """
        :param end_effector: The griplink gripper to build the specification for.
        :param state_type: The state type the specification is labelled with.
        :param configuration: Hardware parameters forwarded to the griplink action
            server; ``None`` uses the default configuration.
        :return: The specification for the given state type, commanding the joint
            state the configuration describes.
        """
        if configuration is None:
            configuration = GriplinkGripperConfiguration()
        return cls(
            end_effector=end_effector,
            joint_state=cls._build_joint_state(end_effector, state_type, configuration),
            configuration=configuration,
        )

    @staticmethod
    def _build_joint_state(
        end_effector: EndEffector,
        state_type: GripperState,
        configuration: GriplinkGripperConfiguration,
    ) -> JointState:
        """
        Build the joint state the specification commands for a state type.

        Preset states command the joint state the robot description declares; subclasses
        override this to derive the joint state from the configuration.

        :param end_effector: The griplink gripper whose connections are commanded.
        :param state_type: The state type the joint state is labelled with.
        :param configuration: The hardware parameters the joint state may derive from.
        :return: The joint state the specification commands.
        """
        return end_effector.get_joint_state_by_type(state_type)


@dataclass(eq=False)
class GriplinkPresetSpecification(GriplinkSpecification):
    """
    A griplink gripper motion driven by a stored grip preset, used for ``Grip``/
    ``Release`` actions; the alternative reads the configuration's
    :attr:`~GriplinkGripperConfiguration.grip_preset`.
    """


@dataclass(eq=False)
class GriplinkFlexSpecification(GriplinkSpecification):
    """
    A griplink gripper motion driven by a commanded opening width, used for
    ``Flexgrip``/``Flexrelease`` actions; the alternative reads the configuration's
    :attr:`~GriplinkGripperConfiguration.grip_position`, :attr:`~grip_force`,
    :attr:`~grip_speed` and :attr:`~grip_acceleration`.
    """

    @staticmethod
    def _build_joint_state(
        end_effector: EndEffector,
        state_type: GripperState,
        configuration: GriplinkGripperConfiguration,
    ) -> JointState:
        """
        Build the joint state a griplink flex motion commands, interpolating between the
        declared open and close states by the configured opening width.

        Without a configured opening width, a ``FLEXCLOSE`` motion commands the declared
        close state and a ``FLEXOPEN`` motion the declared open state.

        :param end_effector: The griplink gripper whose connections are commanded.
        :param state_type: The flex state type the joint state is labelled with.
        :param configuration: Hardware parameters forwarded to the griplink action
            server; its opening width drives the joint state.
        :return: The joint state driving the gripper's connections to the interpolated
            position.
        """
        if configuration.grip_position is not None:
            grip_position = configuration.grip_position
        elif state_type is GripperState.FLEXCLOSE:
            grip_position = FULLY_CLOSED_OPENING_WIDTH_MM
        else:
            grip_position = MAXIMUM_OPENING_WIDTH_MM
        open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)
        close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        fraction = (MAXIMUM_OPENING_WIDTH_MM - grip_position) / MAXIMUM_OPENING_WIDTH_MM
        close_targets = dict(close_state.items())
        target_values = [
            open_target + fraction * (close_targets[connection] - open_target)
            for connection, open_target in open_state.items()
        ]
        return JointState(
            connections=open_state.connections,
            target_values=target_values,
            state_type=state_type,
            name=PrefixedName("flexgrip", prefix=end_effector.name.name),
        )
