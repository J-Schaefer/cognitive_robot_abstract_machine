from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    BallJointConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Color, Cylinder, Scale
from semantic_digital_twin.world_description.shape_collection import (
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(eq=False)
class SimulatedCable(HasRootBody):
    """
    A cable simulated as a chain of rigid capsule links connected by ball joints.

    Uses a chain of bodies with cylinder/capsule geometry connected by ball
    joints to produce physically realistic cable behaviour in MuJoCo.

    .. note::
        Use :meth:`create_in_world` to construct a simulated cable. That method
        creates the link bodies, ball joints, and attachment to the parent body.
    """

    length: float
    """
    Total length of the cable in metres.
    """

    thickness: float
    """
    Radius of each cable link in metres.
    """

    number_of_links: int
    """
    Number of rigid segments that make up the cable chain.
    """

    hanging_from: Body
    """
    The body from which this cable hangs.
    """

    mount_offset_x: float = 0.0
    """
    Offset in x to mount the cable with.
    """

    mount_offset_y: float = 0.0
    """
    Offset in y to mount the cable with.
    """

    height_offset: float = 0.0
    """
    Offset in z to mount the cable with.
    """

    link_bodies: list[Body] = field(kw_only=True)
    """
    The bodies that make up the cable chain, in order from the attachment point to the
    free end.
    """

    link_joints: list[BallJointConnection] = field(kw_only=True)
    """
    The ball joint connections between consecutive link bodies.
    """

    cable_color: Color = field(default_factory=lambda: Color.YELLOW())
    """
    The color of the cable's visual geometry.
    """

    @classmethod
    def create_in_world(
        cls,
        name: PrefixedName,
        world: World,
        hanging_from: Body,
        length: float,
        thickness: float,
        number_of_links: int,
        mount_offset_x: float = 0.0,
        mount_offset_y: float = 0.0,
        height_offset: float = 0.0,
        cable_color: Color = Color.YELLOW(),
    ) -> Self:
        """
        Create a simulated cable with a chain of link bodies connected by ball joints.

        The first link is attached to ``hanging_from`` via a fixed connection. Each
        subsequent link is connected to the previous one via a ball joint with its
        anchor at the far end of the parent link.

        :param name: Base name for the cable body and annotation.
        :param world: The world to register the bodies, connections, and annotation in.
        :param hanging_from: The body the cable is attached to.
        :param length: Total length of the cable in metres.
        :param thickness: Radius of each cable link in metres.
        :param number_of_links: Number of rigid segments.
        :param cable_color: Color of the cable's visual geometry.
        :return: The created SimulatedCable annotation.
        """
        link_length = length / number_of_links
        link_bodies: list[Body] = []
        link_joints: list[BallJointConnection] = []

        for link_index in range(number_of_links):
            link_name = PrefixedName(f"{name.name}_link_{link_index}", name.prefix)
            link_body = Body(name=link_name)
            shape = cls._build_link_geometry(link_body, thickness, link_length)
            link_body.collision = shape
            visual = ShapeCollection(
                [deepcopy(s) for s in shape.shapes], shape.reference_frame
            )
            visual.dye_shapes(cable_color)
            link_body.visual = visual
            world.add_body(link_body)
            link_bodies.append(link_body)

        attachment_transform = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=mount_offset_x,
            y=mount_offset_y,
            z=height_offset,
            reference_frame=hanging_from,
        )
        attachment_connection = FixedConnection(
            parent=hanging_from,
            child=link_bodies[0],
            parent_T_connection_expression=attachment_transform,
        )
        world.add_connection(attachment_connection)

        for link_index in range(number_of_links - 1):
            parent_link = link_bodies[link_index]
            child_link = link_bodies[link_index + 1]
            joint_transform = HomogeneousTransformationMatrix.from_xyz_rpy(
                z=-link_length,
                reference_frame=parent_link,
            )
            ball_joint = BallJointConnection.create_with_dofs(
                world=world,
                parent=parent_link,
                child=child_link,
                parent_T_connection_expression=joint_transform,
            )
            world.add_connection(ball_joint)
            link_joints.append(ball_joint)

        annotation = cls(
            name=name,
            root=link_bodies[0],
            length=length,
            thickness=thickness,
            number_of_links=number_of_links,
            hanging_from=hanging_from,
            link_bodies=link_bodies,
            link_joints=link_joints,
            cable_color=cable_color,
        )
        world.add_semantic_annotation(annotation)
        return annotation

    @staticmethod
    def _build_link_geometry(
        cable_body: Body,
        thickness: float,
        link_length: float,
    ) -> ShapeCollection:
        cylinder = Cylinder(
            width=thickness * 2,
            height=link_length,
        )
        return ShapeCollection([cylinder])
