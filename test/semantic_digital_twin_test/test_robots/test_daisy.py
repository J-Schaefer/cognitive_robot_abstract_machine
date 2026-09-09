from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.gripper_configurations import (
    GripperConfiguration,
    WPGGripperConfiguration,
)


def test_daisy_grippers_carry_wpg_gripper_configuration(daisy_world):
    """
    Both DAiSy end effectors carry a WPG gripper configuration, so the DAiSy gripper
    motion mappings can resolve their hardware parameters from the moved end effector.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.get_end_effectors():
        configuration = end_effector.gripper_configuration
        assert isinstance(configuration, WPGGripperConfiguration)
        assert isinstance(configuration, GripperConfiguration)
