from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.cable_grasp import CableGraspAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="session")
def cable_hanger_world(daisy_world):
    world = daisy_world
    hanger_body = Body(
        name=PrefixedName("hanger"),
        collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
        visual=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
    )

    daisy = world.get_semantic_annotations_by_type(DAiSy)[0]
    hanger_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.4,
        pos_y=0.0,
        pos_z=0.5,
        reference_frame=daisy.root,
    )

    with world.modify_world():
        hanger_connection = Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=hanger_body,
            name=PrefixedName("hanger_connection"),
            parent_T_connection_expression=hanger_pose,
        )
        world.add_connection(hanger_connection)

        cable_annotation = Cable.create_with_new_body_in_world(
            name=PrefixedName("cable"),
            world=world,
            hanging_from=hanger_body,
            length=0.3,
            mount_offset_x=0.0,
            mount_offset_y=0.0,
            height_offset=0.0,
        )

    return world


@pytest.fixture
def immutable_cable_hanger_world(cable_hanger_world):
    world = cable_hanger_world
    state = deepcopy(world.state._data)
    view = world.get_semantic_annotations_by_type(DAiSy)[0]
    context = Context(world, view)
    yield world, view, context
    world.state._data[:] = state
    world.notify_state_change()


@pytest.fixture
def mutable_cable_hanger_world(cable_hanger_world):
    copy_world = deepcopy(cable_hanger_world)
    copy_view = copy_world.get_semantic_annotations_by_type(DAiSy)[0]
    return copy_world, copy_view, Context(copy_world, copy_view)


def test_cable_grasp_attaches_cable_to_end_effector(
    mutable_cable_hanger_world,
):
    world, view, context = mutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    assert len(cable_annotations) == 1
    cable_annotation = cable_annotations[0]

    plan = sequential(
        [
            ParkArmsAction(arm=Arms.BOTH),
            CableGraspAction(
                cable_annotation=cable_annotation,
                grasp_offset=0.1,
                approach_offset=0.1,
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    right_arm = ViewManager.get_arm_view(Arms.RIGHT, view)

    cable_body = cable_annotation.root
    attached_to_left = (
        world.get_connection(
            left_arm.end_effector.tool_frame,
            cable_body,
        )
        is not None
    )
    attached_to_right = (
        world.get_connection(
            right_arm.end_effector.tool_frame,
            cable_body,
        )
        is not None
    )

    assert (
        attached_to_left or attached_to_right
    ), "Cable must be attached to one of the end effectors after grasp"


def test_cable_grasp_chooses_closer_arm_for_scooping(
    immutable_cable_hanger_world,
):
    world, view, context = immutable_cable_hanger_world

    cable_annotations = world.get_semantic_annotations_by_type(Cable)
    cable_annotation = cable_annotations[0]

    action = CableGraspAction(
        cable_annotation=cable_annotation,
        grasp_offset=0.1,
        approach_offset=0.1,
    )
    plan = sequential([ParkArmsAction(arm=Arms.BOTH), action], context=context).plan
    with simulated_robot:
        plan.perform()

    scoop_arm = action._choose_scoop_arm()
    grasp_arm = Arms.RIGHT if scoop_arm == Arms.LEFT else Arms.LEFT

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    right_arm = ViewManager.get_arm_view(Arms.RIGHT, view)

    hanger_pos = action._hanging_point_position().to_np()

    left_tip_pos = (
        left_arm.end_effector.tool_frame.global_transform.to_position().to_np()
    )
    right_tip_pos = (
        right_arm.end_effector.tool_frame.global_transform.to_position().to_np()
    )

    left_distance = float(np.linalg.norm(left_tip_pos - hanger_pos))
    right_distance = float(np.linalg.norm(right_tip_pos - hanger_pos))

    if left_distance <= right_distance:
        assert scoop_arm == Arms.LEFT
    else:
        assert scoop_arm == Arms.RIGHT

    assert scoop_arm != grasp_arm
