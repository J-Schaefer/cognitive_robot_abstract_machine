"""
Tests for pairing the root links and tip links of the interactive marker tool.
"""

import pytest

from giskardpy.exceptions import UnpairedKinematicChainParametersError
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import RootTipPair
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% pairing root links with tip links

ROOT_LINKS = ["map", "arm_mount_left_link"]
"""
Root links of two chains, used to pair against a matching and a mismatched tip list.
"""

TIP_LINKS = ["right_hand_tcp", "left_hand_tcp"]
"""
Tip links belonging to :data:`ROOT_LINKS`, in the same order.
"""


def test_endpoints_are_paired_by_index():
    chains = RootTipPair.pair_up(root_links=ROOT_LINKS, tip_links=TIP_LINKS)

    assert chains == [
        RootTipPair(root_link=ROOT_LINKS[0], tip_link=TIP_LINKS[0]),
        RootTipPair(root_link=ROOT_LINKS[1], tip_link=TIP_LINKS[1]),
    ]


def test_more_root_links_than_tip_links_raises():
    with pytest.raises(UnpairedKinematicChainParametersError):
        RootTipPair.pair_up(root_links=ROOT_LINKS, tip_links=TIP_LINKS[:1])


def test_more_tip_links_than_root_links_raises():
    with pytest.raises(UnpairedKinematicChainParametersError):
        RootTipPair.pair_up(root_links=ROOT_LINKS[:1], tip_links=TIP_LINKS)


# %% resolving link names


@pytest.fixture
def world_whose_robot_has_a_map_link() -> World:
    """
    A world whose root is named ``map``, holding a robot whose description has a ``map``
    link of its own, as DAiSy's does.
    """
    world = World()
    with world.modify_world():
        world_root = Body(name=PrefixedName("map"))
        robot_map = Body(name=PrefixedName("map", prefix="daisy"))
        tool_frame = Body(name=PrefixedName("tool_frame", prefix="daisy"))
        world.add_body(world_root)
        world.add_connection(FixedConnection(parent=world_root, child=robot_map))
        world.add_connection(FixedConnection(parent=robot_map, child=tool_frame))
    return world


def test_a_root_link_named_like_the_world_root_is_the_world_root(
    world_whose_robot_has_a_map_link,
):
    world = world_whose_robot_has_a_map_link
    assert RootTipPair.link_named(world=world, name="map") is world.root


def test_any_other_link_is_looked_up_by_name(world_whose_robot_has_a_map_link):
    world = world_whose_robot_has_a_map_link
    assert RootTipPair.link_named(
        world=world, name="tool_frame"
    ) is world.get_kinematic_structure_entity_by_name("tool_frame")
