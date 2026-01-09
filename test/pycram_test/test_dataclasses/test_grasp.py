import os
from copy import deepcopy

import pytest

from pycram.datastructures.enums import ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import NewGraspDescription
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.robots.tracy import Tracy


@pytest.fixture(scope="session")
def tracy_milk_world(tracy_world):
    tracy_copy = deepcopy(tracy_world)
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "pycram",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    tracy_copy.merge_world(milk_world)
    return tracy_copy


def test_grasp_pose_front(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        world.get_body_by_name("milk.stl"),
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new()

    assert grasp_pose.orientation.to_list() == [0, 0, 0, 1]
    assert grasp_pose.position.to_list() == [0, 0, 0]

    offset_pose = grasp_desc.grasp_pose_new(grasp_edge=True)

    assert grasp_pose.orientation.to_list() == [0, 0, 0, 1]
    assert offset_pose.position.to_list() == pytest.approx([-0.03, 0, 0], abs=0.01)


def test_grasp_pose_right(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        world.get_body_by_name("milk.stl"),
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new()

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, 0.707, 0.707], abs=0.001
    )

    offset_pose = grasp_desc.grasp_pose_new(grasp_edge=True)

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, 0.707, 0.707], abs=0.001
    )
    assert offset_pose.position.to_list() == pytest.approx([-0.03, 0, 0], abs=0.01)


def test_grasp_pose_left(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        world.get_body_by_name("milk.stl"),
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new()

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )

    offset_pose = grasp_desc.grasp_pose_new(grasp_edge=True)

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )
    assert offset_pose.position.to_list() == pytest.approx([-0.03, 0, 0], abs=0.01)


def test_grasp_front_tracy(tracy_milk_world):
    robot_view = Tracy.from_world(tracy_milk_world)

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        tracy_milk_world.get_body_by_name("milk.stl"),
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new()

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )
