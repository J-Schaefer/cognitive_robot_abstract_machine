from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


class CableShape(Enum):
    """
    The cross-sectional shape used for the cable's visual and collision geometry.
    """

    BOX = "box"
    """
    An axis-aligned box with the given thickness along x and y.
    """

    CYLINDER = "cylinder"
    """
    A cylinder with the given thickness as its diameter.
    """


@dataclass(eq=False)
class Cable(HasRootBody):
    """
    A cable hanging from a fixture such as a cable hanger.

    .. note::
        Use :meth:`create_with_new_body_in_world` to construct a cable. That
        method uses ``mount_offset_x``, ``mount_offset_y``, and ``height_offset``
        as the single source of truth for both the kinematic connection and the
        hanging point queried by action designators.
    """

    hanging_from: Body
    """
    The body from which this cable hangs.
    """

    length: float
    """
    The length of the cable in metres.
    """

    mount_offset_x: float = field(default=0.0)
    """
    Offset in metres along the parent body's local X axis for the hanging point.
    """

    mount_offset_y: float = field(default=0.0)
    """
    Offset in metres along the parent body's local Y axis for the hanging point.
    """

    height_offset: float = field(default=0.0)
    """
    Offset in metres along the parent body's local Z axis from the parent origin to the
    hanging point.

    A negative value means the cable hangs below the parent origin.
    """

    cable_shape: CableShape = field(default=CableShape.CYLINDER)
    """
    The cross-sectional shape used for the cable's visual and collision geometry.
    """

    cable_color: Color = field(default_factory=lambda: Color.YELLOW())
    """
    The color of the cable's visual geometry.
    """

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        hanging_from: Body,
        length: float,
        mount_offset_x: float = 0.0,
        mount_offset_y: float = 0.0,
        height_offset: float = 0.0,
        cable_thickness: float = 0.01,
        cable_shape: CableShape = CableShape.CYLINDER,
        cable_color: Color = Color.YELLOW(),
    ) -> Self:
        """
        Create a Cable annotation with a new body whose geometry is a cylinder by
        default.

        The hanging point is at ``(mount_offset_x, mount_offset_y, height_offset)`` in
        the ``hanging_from`` frame. The cable body extends downward from the hanging
        point by ``length``.

        :param name: Name for the cable body and annotation.
        :param world: The world to register the body, connection, and annotation in.
        :param hanging_from: The body the cable is attached to.
        :param length: Overall length of the cable in metres.
        :param mount_offset_x: X offset of the hanging point in the parent frame.
        :param mount_offset_y: Y offset of the hanging point in the parent frame.
        :param height_offset: Z offset of the hanging point in the parent frame.
        :param cable_thickness: Thickness of the cable cross-section in metres.
        :param cable_shape: Cross-sectional shape of the cable geometry.
        :param cable_color: Color of the cable's visual geometry.
        """
        cable_body = Body(name=name)
        collision = cls._build_geometry(
            cable_body, cable_thickness, length, cable_shape
        )
        cable_body.collision = collision
        visual = ShapeCollection([deepcopy(shape) for shape in collision.shapes])
        visual.reference_frame = collision.reference_frame
        visual.dye_shapes(cable_color)
        cable_body.visual = visual

        connection_transform = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=mount_offset_x,
            y=mount_offset_y,
            z=height_offset - length / 2,
            reference_frame=hanging_from,
        )
        connection = FixedConnection(
            parent=hanging_from,
            child=cable_body,
            parent_T_connection_expression=connection_transform,
        )

        world.add_body(cable_body)
        world.add_connection(connection)

        annotation = cls(
            name=name,
            root=cable_body,
            hanging_from=hanging_from,
            length=length,
            mount_offset_x=mount_offset_x,
            mount_offset_y=mount_offset_y,
            height_offset=height_offset,
            cable_shape=cable_shape,
            cable_color=cable_color,
        )
        world.add_semantic_annotation(annotation)

        return annotation

    @staticmethod
    def _build_geometry(
        cable_body: Body,
        cable_thickness: float,
        length: float,
        cable_shape: CableShape,
    ) -> ShapeCollection:
        """
        Build the collision and visual geometry for the cable body.

        :param cable_body: The body the geometry is attached to.
        :param cable_thickness: Thickness of the cable cross-section in metres.
        :param length: Overall length of the cable in metres.
        :param cable_shape: Cross-sectional shape of the cable geometry.
        :return: The shape collection describing the cable geometry.
        """
        scale = Scale(cable_thickness, cable_thickness, length)
        geometry_event = scale.to_simple_event().as_composite_set()
        bounding_boxes = BoundingBoxCollection.from_event(cable_body, geometry_event)
        if cable_shape is CableShape.CYLINDER:
            return ShapeCollection(
                [
                    bounding_box.as_cylinder()
                    for bounding_box in bounding_boxes.bounding_boxes
                ],
                bounding_boxes.reference_frame,
            )
        return bounding_boxes.as_shapes()
