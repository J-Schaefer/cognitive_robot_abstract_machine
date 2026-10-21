from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy
from numpy.typing import NDArray
from typing_extensions import Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.exceptions import (
    ArcLengthOutsideCableError,
    TooFewCableSegmentsError,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

# %% polyline geometry


@dataclass
class Polyline:
    """
    A chain of straight segments through a sequence of corners.
    """

    corners: NDArray[numpy.float64]
    """
    The corners, of shape ``(number of corners, 3)``.
    """

    @cached_property
    def arc_lengths(self) -> NDArray[numpy.float64]:
        """
        The distance of each corner from the first one along the polyline.
        """
        segment_lengths = numpy.linalg.norm(numpy.diff(self.corners, axis=0), axis=1)
        return numpy.concatenate([[0.0], numpy.cumsum(segment_lengths)])

    @property
    def length(self) -> float:
        """
        The length of the polyline.
        """
        return float(self.arc_lengths[-1])

    def point_at(self, arc_length: float) -> NDArray[numpy.float64]:
        """
        :param arc_length: The distance along the polyline from its first corner.
        :return: The point at that distance.
        """
        return numpy.array(
            [
                numpy.interp(arc_length, self.arc_lengths, self.corners[:, axis])
                for axis in range(3)
            ]
        )

    def resampled(self, number_of_segments: int) -> Polyline:
        """
        :param number_of_segments: How many segments the result has.
        :return: A polyline along this one whose corners are spaced evenly.
        """
        return Polyline(
            numpy.array(
                [
                    self.point_at(arc_length)
                    for arc_length in numpy.linspace(
                        0.0, self.length, number_of_segments + 1
                    )
                ]
            )
        )


# %% material


@dataclass
class RodMaterial:
    """
    The isotropic, linear elastic material a cable is made of.
    """

    young_modulus: float = 5.0e6
    """
    Young's modulus, in pascal.
    """

    poisson_ratio: float = 0.45
    """
    Poisson's ratio, dimensionless.
    """

    density: float = 1200.0
    """
    Mass density, in kilograms per cubic metre.
    """

    stiffness_damping: float = 1.0e-3
    """
    Rayleigh damping coefficient proportional to the stiffness, in seconds.
    """

    friction_coefficient: float = 0.9
    """
    Coulomb friction coefficient of the cable's surface.
    """

    friction_falloff_velocity: float = 1.0e-3
    """
    Sliding speed, in metres per second, at which friction reaches its full Coulomb
    value; below it, friction grows in proportion to the speed, so a held cable creeps
    slowly instead of sticking.
    """

    @property
    def shear_modulus(self) -> float:
        """
        The shear modulus, in pascal.
        """
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))


# %% cable


@dataclass(eq=False)
class DeformableCable(SemanticAnnotation):
    """
    A cable with a circular cross section that bends, twists and stretches, described by
    its centreline.

    The cable has no bodies: its shape is the polyline of :attr:`nodes`, which a
    deformable body simulation keeps up to date.
    """

    radius: float = field(kw_only=True)
    """
    Radius of the cross section, in metres.
    """

    material: RodMaterial = field(default_factory=RodMaterial, kw_only=True)
    """
    What the cable is made of.
    """

    nodes: list[Point3] = field(default_factory=list, kw_only=True)
    """
    The centreline from one end of the cable to the other, in the frame of the world
    root.
    """

    color: Color = field(default_factory=Color.WHITE, kw_only=True)
    """
    The colour the cable is drawn in.
    """

    @classmethod
    def along_polyline(
        cls,
        name: PrefixedName,
        polyline: list[Point3],
        number_of_segments: int,
        radius: float,
        material: RodMaterial | None = None,
        color: Color | None = None,
    ) -> Self:
        """
        Create a cable laid along a polyline, with nodes spaced evenly along it.

        :param name: The name of the cable.
        :param polyline: The corners the cable runs through, from one end to the other,
            in the frame of the world root.
        :param number_of_segments: How many straight segments the centreline has.
        :param radius: Radius of the cross section, in metres.
        :param material: What the cable is made of; the default material if ``None``.
        :param color: The colour the cable is drawn in; white if ``None``.
        :return: The cable, not yet part of a world.
        :raises TooFewCableSegmentsError: If ``number_of_segments`` is below one.
        """
        if number_of_segments < 1:
            raise TooFewCableSegmentsError(number_of_segments=number_of_segments)
        corners = Polyline(numpy.array([point.to_np()[:3] for point in polyline]))
        reference_frame = polyline[0].reference_frame
        return cls(
            name=name,
            radius=radius,
            material=material if material is not None else RodMaterial(),
            color=color if color is not None else Color.WHITE(),
            nodes=[
                Point3.from_iterable(corner, reference_frame=reference_frame)
                for corner in corners.resampled(number_of_segments).corners
            ],
        )

    @property
    def centreline(self) -> Polyline:
        """
        The polyline through the nodes.
        """
        return Polyline(numpy.array([node.to_np()[:3] for node in self.nodes]))

    @property
    def length(self) -> float:
        """
        The length of the centreline, in metres.
        """
        return self.centreline.length

    @property
    def cross_section_area(self) -> float:
        """
        Area of the cross section, in square metres.
        """
        return math.pi * self.radius**2

    @property
    def second_moment_of_area(self) -> float:
        """
        Second moment of area of the cross section about a diameter, in metres to the
        fourth.
        """
        return math.pi * self.radius**4 / 4.0

    @property
    def axial_stiffness(self) -> float:
        """
        Resistance to stretching, in newtons.
        """
        return self.material.young_modulus * self.cross_section_area

    @property
    def bending_stiffness(self) -> float:
        """
        Resistance to bending, in newton square metres.
        """
        return self.material.young_modulus * self.second_moment_of_area

    @property
    def torsional_stiffness(self) -> float:
        """
        Resistance to twisting, in newton square metres; the polar moment of a circle is
        twice its second moment of area.
        """
        return self.material.shear_modulus * 2.0 * self.second_moment_of_area

    @property
    def linear_density(self) -> float:
        """
        Mass per length, in kilograms per metre.
        """
        return self.material.density * self.cross_section_area

    @property
    def rotational_inertia_per_length(self) -> float:
        """
        Moment of inertia about the centreline per length, in kilogram metres.
        """
        return self.material.density * 2.0 * self.second_moment_of_area

    def point_at_arc_length(self, arc_length: float) -> Point3:
        """
        The point on the centreline a given distance from its first node.

        :param arc_length: The distance along the centreline, in metres.
        :return: The point, in the frame of the nodes.
        :raises ArcLengthOutsideCableError: If the distance is negative or longer than
            the cable.
        """
        centreline = self.centreline
        if not 0.0 <= arc_length <= centreline.length:
            raise ArcLengthOutsideCableError(
                arc_length=arc_length, cable_length=centreline.length
            )
        return Point3.from_iterable(
            centreline.point_at(arc_length),
            reference_frame=self.nodes[0].reference_frame,
        )
