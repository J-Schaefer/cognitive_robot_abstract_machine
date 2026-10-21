import math

import numpy
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.deformable_cable import (
    DeformableCable,
    Polyline,
    RodMaterial,
)
from semantic_digital_twin.semantic_annotations.exceptions import (
    ArcLengthOutsideCableError,
    TooFewCableSegmentsError,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


@pytest.fixture
def l_shaped_cable() -> DeformableCable:
    """
    A cable running 0.3 m down and then 0.2 m along x, in ten segments.
    """
    return DeformableCable.along_polyline(
        name=PrefixedName("cable"),
        polyline=[Point3(0, 0, 0.3), Point3(0, 0, 0), Point3(0.2, 0, 0)],
        number_of_segments=10,
        radius=0.0025,
    )


# %% polyline


def test_a_polyline_is_as_long_as_its_segments():
    polyline = Polyline(
        numpy.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 2.0]])
    )

    assert polyline.length == pytest.approx(7.0)


def test_a_resampled_polyline_spaces_its_corners_evenly():
    polyline = Polyline(
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    )

    resampled = polyline.resampled(number_of_segments=6)

    segment_lengths = numpy.diff(resampled.arc_lengths)
    numpy.testing.assert_allclose(segment_lengths, polyline.length / 6)


# %% cable geometry


def test_a_cable_along_a_polyline_starts_and_ends_at_its_ends(l_shaped_cable):
    numpy.testing.assert_allclose(l_shaped_cable.nodes[0].to_np()[:3], [0, 0, 0.3])
    numpy.testing.assert_allclose(l_shaped_cable.nodes[-1].to_np()[:3], [0.2, 0, 0])


def test_a_cable_along_a_polyline_has_one_node_more_than_segments(l_shaped_cable):
    assert len(l_shaped_cable.nodes) == 11


def test_a_cable_along_a_polyline_is_as_long_as_the_polyline(l_shaped_cable):
    assert l_shaped_cable.length == pytest.approx(0.5)


def test_a_point_along_a_cable_lies_that_far_along_its_centreline(l_shaped_cable):
    point = l_shaped_cable.point_at_arc_length(0.4)

    numpy.testing.assert_allclose(point.to_np()[:3], [0.1, 0, 0], atol=1e-12)


def test_no_point_lies_beyond_the_end_of_a_cable(l_shaped_cable):
    with pytest.raises(ArcLengthOutsideCableError):
        l_shaped_cable.point_at_arc_length(0.6)


def test_a_cable_needs_at_least_one_segment():
    with pytest.raises(TooFewCableSegmentsError):
        DeformableCable.along_polyline(
            name=PrefixedName("cable"),
            polyline=[Point3(0, 0, 0), Point3(1, 0, 0)],
            number_of_segments=0,
            radius=0.0025,
        )


# %% stiffness of a circular cross section


def test_the_bending_stiffness_of_a_cable_is_that_of_a_round_beam(l_shaped_cable):
    material = l_shaped_cable.material
    radius = l_shaped_cable.radius

    assert l_shaped_cable.bending_stiffness == pytest.approx(
        material.young_modulus * math.pi * radius**4 / 4
    )


def test_a_round_cable_resists_twisting_with_twice_its_second_moment_of_area():
    cable = DeformableCable(
        name=PrefixedName("cable"),
        radius=0.002,
        material=RodMaterial(young_modulus=2.6e6, poisson_ratio=0.3),
    )

    assert cable.torsional_stiffness == pytest.approx(
        1.0e6 * 2 * cable.second_moment_of_area
    )


# %% the cable in a world


def test_two_cables_of_different_names_are_both_part_of_a_world():
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
        for name in ("first", "second"):
            world.add_semantic_annotation(
                DeformableCable.along_polyline(
                    name=PrefixedName(name),
                    polyline=[Point3(0, 0, 0), Point3(1, 0, 0)],
                    number_of_segments=2,
                    radius=0.0025,
                )
            )

    assert len(world.get_semantic_annotations_by_type(DeformableCable)) == 2
