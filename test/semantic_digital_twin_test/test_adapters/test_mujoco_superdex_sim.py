from datetime import timedelta

import numpy
import pytest

pytest.importorskip("superdex.physics")

from semantic_digital_twin.adapters.superdex.mujoco_cable_simulation import (
    MujocoBodyPoses,
    MujocoSuperdexSim,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.deformable_cable import DeformableCable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% scene


@pytest.fixture
def table_and_cable_simulation():
    """
    A running simulation of a table, turned about its vertical axis, with its top at
    0.4 m and a cable lying level 0.05 m above it.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        table = Body(
            name=PrefixedName("table"),
            collision=ShapeCollection([Box(scale=Scale(0.6, 0.4, 0.4))]),
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=table,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.1, z=0.2, yaw=0.3
                ),
            )
        )
        world.add_semantic_annotation(
            DeformableCable.along_polyline(
                name=PrefixedName("cable"),
                polyline=[
                    Point3(-0.1, 0, 0.45, reference_frame=root),
                    Point3(0.2, 0, 0.45, reference_frame=root),
                ],
                number_of_segments=15,
                radius=0.0025,
            )
        )
    simulation = MujocoSuperdexSim(world=world, colliding_bodies=[table], headless=True)
    simulation.start_stepped_simulation()
    yield world, table, simulation
    simulation.stop_simulation()


# %% tests


def test_mujoco_puts_a_fixed_body_where_the_world_has_it(table_and_cable_simulation):
    world, table, simulation = table_and_cable_simulation

    numpy.testing.assert_allclose(
        MujocoBodyPoses(simulator=simulation.simulator).world_T_body(table),
        world.compute_forward_kinematics_np(world.root, table),
        atol=1e-6,
    )


def test_stepping_the_simulation_drops_the_cable_onto_the_table(
    table_and_cable_simulation,
):
    world, table, simulation = table_and_cable_simulation
    [cable] = world.get_semantic_annotations_by_type(DeformableCable)

    for _ in range(50):
        simulation.step_simulation(timedelta(milliseconds=20))

    table_top = (
        table.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .max_z
    )
    assert cable.centreline.corners[:, 2].min() == pytest.approx(
        table_top + cable.radius, abs=1e-3
    )
