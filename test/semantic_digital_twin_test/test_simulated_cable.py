from __future__ import annotations

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.simulated_cable import SimulatedCable
from semantic_digital_twin.world_description.connections import BallJointConnection
from semantic_digital_twin.world_description.geometry import Color, Cylinder
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World


class TestSimulatedCable:
    """
    Tests for SimulatedCable annotation creation.
    """

    @pytest.fixture
    def empty_world(self):
        """
        An empty world with a hanger body.
        """
        world = World()
        hanger = Body(name=PrefixedName("hanger"))
        with world.modify_world():
            world.add_body(hanger)
        return world, hanger

    def test_creates_link_bodies(self, empty_world):
        """
        A simulated cable creates the correct number of link bodies.
        """
        world, hanger = empty_world
        with world.modify_world():
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=hanger,
                length=0.3,
                thickness=0.005,
                number_of_links=5,
            )
        assert len(cable.link_bodies) == 5
        for i, link in enumerate(cable.link_bodies):
            assert isinstance(link, Body)
            expected_name = f"cable_link_{i}"
            assert link.name.name == expected_name
            assert len(link.collision.shapes) == 1
            shape = link.collision.shapes[0]
            assert isinstance(shape, Cylinder)
            assert shape.height == pytest.approx(0.06)

    def test_creates_ball_joints_between_links(self, empty_world):
        """
        Consecutive links are connected by ball joints.
        """
        world, hanger = empty_world
        with world.modify_world():
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=hanger,
                length=0.3,
                thickness=0.005,
                number_of_links=5,
            )
        assert len(cable.link_joints) == 4
        for joint in cable.link_joints:
            assert isinstance(joint, BallJointConnection)

    def test_first_link_attached_to_hanger(self, empty_world):
        """
        The first link is attached to the hanger body via a fixed connection.
        """
        world, hanger = empty_world
        with world.modify_world():
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=hanger,
                length=0.3,
                thickness=0.005,
                number_of_links=5,
            )
        first_link = cable.link_bodies[0]
        parent = first_link.parent_connection.parent
        assert parent == hanger

    def test_cable_visual_is_colored(self, empty_world):
        """
        The cable visual geometry is colored.
        """
        world, hanger = empty_world
        with world.modify_world():
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=hanger,
                length=0.3,
                thickness=0.005,
                number_of_links=3,
                cable_color=Color.BLUE(),
            )
        for link in cable.link_bodies:
            for vis_shape in link.visual.shapes:
                assert vis_shape.color.R == 0.0
                assert vis_shape.color.B == 1.0

    def test_link_length_computed_from_total_length(self, empty_world):
        """Each link has length = total_length / number_of_links."""
        world, hanger = empty_world
        with world.modify_world():
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=hanger,
                length=0.4,
                thickness=0.01,
                number_of_links=8,
            )
        for link in cable.link_bodies:
            shape = link.collision.shapes[0]
            assert shape.height == pytest.approx(0.4 / 8)

    def test_root_is_first_link(self, empty_world):
        """
        The root of the annotation is the first link body.
        """
        world, hanger = empty_world
        with world.modify_world():
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=hanger,
                length=0.3,
                thickness=0.005,
                number_of_links=5,
            )
        assert cable.root == cable.link_bodies[0]
