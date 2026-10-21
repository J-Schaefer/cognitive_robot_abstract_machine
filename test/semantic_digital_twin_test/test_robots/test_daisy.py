from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkFlexConfiguration,
    GriplinkPresetConfiguration,
)


def test_daisy_grippers_build_griplink_configurations(daisy_world):
    """
    Both DAiSy end effectors build griplink-specific configurations from their declared
    states, so generic actions produce robot-appropriate configurations without naming
    the robot.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.all_end_effectors:
        open_configuration = end_effector.default_configuration(GripperState.OPEN)
        close_configuration = end_effector.default_configuration(GripperState.CLOSE)
        flex_close_configuration = end_effector.default_configuration(
            GripperState.FLEXCLOSE
        )

        assert isinstance(open_configuration, GriplinkPresetConfiguration)
        assert isinstance(close_configuration, GriplinkPresetConfiguration)
        assert isinstance(flex_close_configuration, GriplinkFlexConfiguration)


# %% physical simulation


def test_daisy_declares_a_servo_for_every_arm_and_gripper_joint(daisy_world):
    """
    A physical simulation drives every joint of both arms and the driving finger joint
    of both grippers with a position servo.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    servoed = {
        degree_of_freedom
        for actuator in daisy_world.actuators
        for degree_of_freedom in actuator.dofs
    }

    for part in (*daisy.all_arms, *daisy.all_end_effectors):
        for connection in part.active_connections:
            assert connection.raw_dof in servoed
