from __future__ import annotations

from math import pi

import numpy as np
import pytest

from coraplex.robot_plans.actions.core.cable_actions import (
    CableRehangAction,
    _hanger_axes,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


@pytest.fixture
def rotated_hanger_world():
    """
    A world with a cable hanger rotated by 90 degrees yaw relative to the world.

    The post body is the world root and the hanger is attached to it with the rotated
    transform, mirroring the daisy cable demo setup. A hanger-frame offset is therefore
    distinguishable from a world-frame offset with the same components.
    """
    world = World()
    cable_post = Body(name=PrefixedName("cable_post"))
    hanger_body = Body(name=PrefixedName("hanger"))

    with world.modify_world():
        world.add_body(cable_post)
        world.add_body(hanger_body)
        world.add_connection(
            FixedConnection(
                parent=cable_post,
                child=hanger_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.6,
                    y=0.4,
                    z=1.0,
                    yaw=pi / 2,
                    reference_frame=cable_post,
                ),
            )
        )
        cable_annotation = Cable.create_with_new_body_in_world(
            name=PrefixedName("cable"),
            world=world,
            hanging_from=hanger_body,
            length=0.3,
        )

    return world, hanger_body, cable_annotation


# %% hanging point offsets


@pytest.mark.parametrize(
    ("approach_direction", "approach_sign"),
    [(0, 1), (0, -1), (1, 1), (1, -1)],
)
def test_rehang_hanging_point_applies_offsets_along_hanger_axes(
    rotated_hanger_world, approach_direction: int, approach_sign: int
):
    """
    The side, front, and up offsets of the rehang hanging point must be applied along
    the hanger's own axes.

    The hanger frame is rotated relative to the world, so an offset interpreted in the
    wrong frame lands on a different world axis and fails this test.
    """
    world, hanger_body, cable_annotation = rotated_hanger_world

    action = CableRehangAction(
        cable_annotation=cable_annotation,
        hanger_body=hanger_body,
        side_offset=0.05,
        front_offset=0.07,
        up_offset=0.03,
        approach_direction=approach_direction,
        approach_sign=approach_sign,
    )

    front_world, side_world, up_world = _hanger_axes(
        hanger_body.global_transform,
        action.approach_direction,
        action.approach_sign,
    )
    hanger_position = hanger_body.global_transform.to_position().to_np()[:3]
    expected = (
        hanger_position
        + front_world * action.front_offset
        + side_world * action.side_offset
        + up_world * action.up_offset
    )

    actual = action._hanging_point_position().to_np()[:3]

    np.testing.assert_allclose(actual, expected, atol=1e-9)
