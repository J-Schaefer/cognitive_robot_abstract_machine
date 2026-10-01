from __future__ import annotations

from copy import deepcopy

import pytest
from coraplex.datastructures.dataclasses import Context
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

    hanger_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.4,
        pos_y=0.0,
        pos_z=0.5,
        reference_frame=world.root,
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

        Cable.create_with_new_body_in_world(
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
