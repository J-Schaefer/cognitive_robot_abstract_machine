"""
Tests of the DAiSy robot description.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyLink
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkFlexConfiguration,
    GriplinkPresetConfiguration,
)

# %% griplink configurations


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


# %% collision avoidance between a forearm and the grippers


def _buffer_zones_of(world, forearm: DAiSyLink) -> dict[str, float]:
    """
    :return: The buffer zone of every checked pair of the forearm with a gripper link,
        by the gripper link's name.
    """
    collision_manager = world.collision_manager
    collision_manager.update_collision_matrix()
    forearm_body = world.get_body_by_name(forearm)
    return {
        other.name.name: collision_manager.get_buffer_zone_distance(forearm_body, other)
        for check in collision_manager.collision_matrix.collision_checks
        if forearm_body in (check.body_a, check.body_b)
        for other in (check.body_a, check.body_b)
        if other is not forearm_body and "gripper" in other.name.name
    }


@pytest.mark.parametrize(
    "forearm, own_gripper_links, links_the_demo_bends_close",
    [
        (
            DAiSyLink.LEFT_FOREARM,
            DAiSyLink.left_gripper_links(),
            {DAiSyLink.LEFT_GRIPPER_BASE, DAiSyLink.LEFT_GRIPPER_WRIST_CYLINDER},
        ),
        (
            DAiSyLink.RIGHT_FOREARM,
            DAiSyLink.right_gripper_links(),
            {DAiSyLink.RIGHT_GRIPPER_BASE, DAiSyLink.RIGHT_GRIPPER_WRIST_CYLINDER},
        ),
    ],
)
def test_forearm_keeps_the_small_buffer_zone_to_its_own_gripper(
    daisy_world,
    forearm: DAiSyLink,
    own_gripper_links: list[DAiSyLink],
    links_the_demo_bends_close: set[DAiSyLink],
):
    buffer_zones = _buffer_zones_of(daisy_world, forearm)

    own_gripper_buffer_zones = {
        name: distance
        for name, distance in buffer_zones.items()
        if name in own_gripper_links
    }
    assert links_the_demo_bends_close <= own_gripper_buffer_zones.keys()
    assert set(own_gripper_buffer_zones.values()) == {
        DAiSy.forearm_to_own_gripper_buffer_zone_distance
    }


def test_right_forearm_keeps_the_larger_buffer_zone_to_the_left_gripper(daisy_world):
    buffer_zones = _buffer_zones_of(daisy_world, DAiSyLink.RIGHT_FOREARM)

    assert (
        buffer_zones[DAiSyLink.LEFT_GRIPPER_SIDE_CYLINDER]
        == DAiSy.forearm_to_other_gripper_buffer_zone_distance
    )


# %% action topics


@pytest.mark.parametrize(
    "state", [GripperState.OPEN, GripperState.CLOSE, GripperState.FLEXCLOSE]
)
def test_daisy_grippers_serve_each_state_under_the_topic_of_its_griplink_endpoint(
    daisy_world, state
):
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.all_end_effectors:
        assert (
            end_effector.action_topic_serving(state)
            == end_effector.griplink_endpoint(state).action_topic
        )


def test_daisy_grippers_serve_no_state_without_a_griplink_endpoint(daisy_world):
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.all_end_effectors:
        assert end_effector.action_topic_serving(None) is None


# %% importing without ROS


def test_daisy_is_importable_without_griplink_interfaces():
    """
    The DAiSy robot description loads in a process without the ROS griplink action
    package, so code and tests that never command the real grippers run without ROS.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "daisy_import_without_griplink_interfaces.py"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == DAiSy.__name__
