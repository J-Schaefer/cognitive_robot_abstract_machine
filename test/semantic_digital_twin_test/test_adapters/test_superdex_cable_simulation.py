from datetime import timedelta

import numpy
import pytest

pytest.importorskip("superdex.physics")

from semantic_digital_twin.adapters.superdex.cable_simulation import (
    BodyMotion,
    KinematicBodyPoses,
    SuperdexCableSimulation,
)
from semantic_digital_twin.adapters.superdex.exceptions import (
    BodyWithoutCollisionShapesError,
    CableNodeOutOfRangeError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.deformable_cable import DeformableCable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% scene


TABLE_THICKNESS = 0.06
"""
How thick the table of the test scene is, in metres.
"""


def build_cable_world(cable_ends: list[tuple[float, float, float]]):
    """
    A table whose top is at height zero, two box fingers on free connections, and a
    straight cable between two points.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        table = Body(
            name=PrefixedName("table"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, TABLE_THICKNESS))]),
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=table,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-TABLE_THICKNESS / 2
                ),
            )
        )
        finger_connections = []
        for name in ("left_finger", "right_finger"):
            finger = Body(
                name=PrefixedName(name),
                collision=ShapeCollection([Box(scale=Scale(0.02, 0.008, 0.03))]),
            )
            connection = Connection6DoF.create_with_dofs(
                world=world, parent=root, child=finger
            )
            world.add_connection(connection)
            finger_connections.append(connection)
        radius = 0.0025
        cable = DeformableCable.along_polyline(
            name=PrefixedName("cable"),
            polyline=[Point3(*end, reference_frame=root) for end in cable_ends],
            number_of_segments=25,
            radius=radius,
        )
        world.add_semantic_annotation(cable)
    return world, table, finger_connections, cable


@pytest.fixture
def cable_world():
    """
    The scene with the cable hanging from 0.3 m down to 0.05 m above the table.
    """
    return build_cable_world([(0, 0, 0.3), (0, 0, 0.05)])


@pytest.fixture
def lying_cable_world():
    """
    The scene with the cable lying level, 0.05 m above the table.
    """
    return build_cable_world([(0, 0, 0.05), (0.25, 0, 0.05)])


def place_fingers(
    finger_connections: list[Connection6DoF], half_gap: float, height: float
) -> None:
    """
    Stand the fingers on either side of the cable's hanging part, their facing sides
    ``half_gap`` from its axis.
    """
    finger_half_thickness = 0.004
    for connection, side in zip(finger_connections, (1.0, -1.0)):
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=side * (half_gap + finger_half_thickness),
            z=height,
            reference_frame=connection.parent,
        )


def simulation_of(world, table, finger_connections) -> SuperdexCableSimulation:
    return SuperdexCableSimulation(
        world=world,
        colliding_bodies=[table, *(c.child for c in finger_connections)],
        body_poses=KinematicBodyPoses(world=world),
    )


def step_for(simulation: SuperdexCableSimulation, duration: timedelta) -> None:
    for _ in range(round(duration / timedelta(milliseconds=20))):
        simulation.step(timedelta(milliseconds=20))


# %% building


def test_a_cable_starts_on_the_nodes_it_was_laid_along(cable_world):
    world, table, finger_connections, cable = cable_world
    laid_along = cable.centreline.corners.copy()
    place_fingers(finger_connections, half_gap=0.02, height=0.15)

    simulation = simulation_of(world, table, finger_connections)

    numpy.testing.assert_allclose(
        simulation.cables[0].node_positions, laid_along, atol=1e-6
    )


def test_a_body_without_collision_shapes_cannot_collide_with_a_cable(cable_world):
    world, _, _, _ = cable_world

    with pytest.raises(BodyWithoutCollisionShapesError):
        SuperdexCableSimulation(
            world=world,
            colliding_bodies=[world.root],
            body_poses=KinematicBodyPoses(world=world),
        )


# %% contact


def test_a_dropped_cable_comes_to_rest_on_the_table_on_its_surface(
    lying_cable_world,
):
    world, table, finger_connections, cable = lying_cable_world
    place_fingers(finger_connections, half_gap=0.02, height=0.15)
    simulation = simulation_of(world, table, finger_connections)

    step_for(simulation, timedelta(seconds=2))

    assert cable.centreline.corners[:, 2].min() == pytest.approx(cable.radius, abs=1e-3)


def test_fingers_squeezing_a_cable_lift_it_by_friction(cable_world):
    world, table, finger_connections, cable = cable_world
    height = 0.15
    place_fingers(finger_connections, half_gap=0.02, height=height)
    simulation = simulation_of(world, table, finger_connections)
    pin = simulation.pin(cable, 0, world.root)
    step_for(simulation, timedelta(seconds=0.4))
    squeezed_half_gap = cable.radius - 0.0007
    for fraction in numpy.linspace(0.0, 1.0, 25):
        place_fingers(
            finger_connections,
            half_gap=0.02 + fraction * (squeezed_half_gap - 0.02),
            height=height,
        )
        simulation.step(timedelta(milliseconds=20))
    simulation.release(pin)
    lift = 0.1
    for fraction in numpy.linspace(0.0, 1.0, 50):
        place_fingers(
            finger_connections,
            half_gap=squeezed_half_gap,
            height=height + fraction * lift,
        )
        simulation.step(timedelta(milliseconds=20))

    between_the_fingers = numpy.array([0.0, 0.0, height + lift])
    nodes = cable.centreline.corners
    assert numpy.linalg.norm(nodes - between_the_fingers, axis=1).min() < 0.015


# %% pins


def test_a_pinned_node_stays_where_it_was_pinned(cable_world):
    world, table, finger_connections, cable = cable_world
    place_fingers(finger_connections, half_gap=0.02, height=0.15)
    simulation = simulation_of(world, table, finger_connections)
    pinned_at = cable.centreline.corners[0].copy()
    simulation.pin(cable, 0, world.root)

    step_for(simulation, timedelta(seconds=0.5))

    numpy.testing.assert_allclose(cable.centreline.corners[0], pinned_at, atol=5e-3)


def test_a_released_node_falls(lying_cable_world):
    world, table, finger_connections, cable = lying_cable_world
    place_fingers(finger_connections, half_gap=0.02, height=0.15)
    simulation = simulation_of(world, table, finger_connections)
    simulation.release(simulation.pin(cable, 0, world.root))

    step_for(simulation, timedelta(seconds=0.5))

    assert cable.centreline.corners[0][2] == pytest.approx(cable.radius, abs=1e-3)


def test_a_node_a_cable_does_not_have_cannot_be_pinned(cable_world):
    world, table, finger_connections, cable = cable_world
    place_fingers(finger_connections, half_gap=0.02, height=0.15)
    simulation = simulation_of(world, table, finger_connections)

    with pytest.raises(CableNodeOutOfRangeError):
        simulation.pin(cable, len(cable.nodes), world.root)


# %% body motion


def test_a_body_halfway_through_a_motion_is_halfway_there_and_halfway_turned():
    start = numpy.eye(4)
    end = numpy.eye(4)
    end[:3, 3] = [0.2, 0.0, 0.4]
    end[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    halfway = BodyMotion(start=start, end=end).pose_at(0.5)

    angle = numpy.pi / 4
    numpy.testing.assert_allclose(halfway[:3, 3], [0.1, 0.0, 0.2])
    numpy.testing.assert_allclose(
        halfway[:3, :3],
        [
            [numpy.cos(angle), -numpy.sin(angle), 0.0],
            [numpy.sin(angle), numpy.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-12,
    )
