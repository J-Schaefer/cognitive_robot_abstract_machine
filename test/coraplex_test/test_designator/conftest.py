from __future__ import annotations

from copy import deepcopy

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.cable import Cable
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from ...plan_running import robot_extensions
from ..world_snapshot import WorldSnapshot


@pytest.fixture(scope="session")
def cable_hanger_world(daisy_world):
    """
    A copy of the DAiSy world with a cable hanging from a hanger in front of the robot.
    """
    world = deepcopy(daisy_world)
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
    """
    The shared cable hanger world, its DAiSy and the context extensions of its plans,
    returned to its initial model and state after the test.
    """
    snapshot = WorldSnapshot.capture(cable_hanger_world)
    daisy = cable_hanger_world.get_semantic_annotations_by_type(DAiSy)[0]
    yield cable_hanger_world, daisy, robot_extensions(daisy)
    snapshot.restore()


@pytest.fixture
def mutable_cable_hanger_world(cable_hanger_world):
    """
    A copy of the cable hanger world, its DAiSy and the context extensions of its plans.
    """
    world = deepcopy(cable_hanger_world)
    daisy = world.get_semantic_annotations_by_type(DAiSy)[0]
    return world, daisy, robot_extensions(daisy)
