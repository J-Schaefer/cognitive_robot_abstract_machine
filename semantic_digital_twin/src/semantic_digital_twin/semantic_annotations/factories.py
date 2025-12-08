from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import reduce
from operator import or_

from random_events.product_algebra import *
from typing_extensions import Type

from krrood.entity_query_language.entity import (
    let,
    entity,
    not_,
    in_,
    for_all,
)
from krrood.entity_query_language.quantify_entity import an
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import (
    TransformationMatrix,
    Vector3,
    Point3,
)
from ..world import World
from ..world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from ..world_description.degree_of_freedom import DegreeOfFreedom
from ..world_description.geometry import Scale
from ..world_description.shape_collection import BoundingBoxCollection, ShapeCollection
from ..world_description.world_entity import (
    Body,
    Region,
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from ..semantic_annotations.semantic_annotations import (
        Handle,
        Dresser,
        Drawer,
        Door,
        Wall,
        DoubleDoor,
        Room,
        Floor,
    )


@dataclass
class HasDoorLikeFactories(ABC):
    """
    Mixin for factories receiving multiple DoorLikeFactories.
    """

    door_like_factory_configs: List[DoorLikeConfigForParentFactory] = field(
        default_factory=list, hash=False
    )
    """
    The door factories used to create the doors.
    """

    def add_doorlike_semantic_annotation_to_world(
        self,
        parent_world: World,
    ):
        """
        Adds door-like semantic annotations to the parent world.
        """
        for config in self.door_like_factory_configs:
            match config:
                case DoorConfigForParentFactory():
                    self._add_door_to_world(
                        door_factory=config.factory_instance,
                        parent_T_door=config.parent_T_child,
                        opening_axis=config.hinge_axis,
                        parent_world=parent_world,
                    )
                case DoubleDoorConfigForParentFactory():
                    self._add_double_door_to_world(
                        door_factory=config.factory_instance,
                        parent_T_double_door=config.parent_T_child,
                        parent_world=parent_world,
                    )

    def _add_double_door_to_world(
        self,
        door_factory: DoubleDoorFactory,
        parent_T_double_door: TransformationMatrix,
        parent_world: World,
    ):
        """
        Adds a double door to the parent world by extracting the doors from the double door world, moving the
        relevant bodies and semantic annotations to the parent world.
        :param door_factory: The factory used to create the double door.
        :param parent_T_double_door: The transformation matrix defining the double door's position and orientation relative
        to the parent world.
        :param parent_world: The world to which the double door will be added.
        """
        parent_root = parent_world.root
        double_door_world = door_factory.create()
        double_door = double_door_world.get_semantic_annotations_by_type(DoubleDoor)[0]
        doors = [double_door.left_door, double_door.right_door]
        new_worlds = []
        new_connections = []
        new_dofs = []
        for door in doors:
            new_world, new_connection, new_dof = self._move_door_into_new_world(
                parent_root, door, parent_T_double_door
            )
            new_worlds.append(new_world)
            new_connections.append(new_connection)
            new_dofs.append(new_dof)

        with parent_world.modify_world():

            with double_door_world.modify_world():

                for new_door_world, new_parent_C_left, new_dof in zip(
                    new_worlds, new_connections, new_dofs
                ):
                    parent_world.add_degree_of_freedom(new_dof)
                    parent_world.merge_world(new_door_world, new_parent_C_left)

                double_door_world.remove_semantic_annotation(double_door)
                parent_world.add_semantic_annotation(double_door)

    @staticmethod
    def _move_door_into_new_world(
        new_parent: KinematicStructureEntity,
        door: Door,
        parent_T_double_door: TransformationMatrix,
    ):
        """
        Move a door from a double door world into a new world with a revolute connection.

        :param new_parent: Entity that will be the new parent of the door
        :param door: The door to be moved into a new world
        :param parent_T_double_door: Original transform from parent to double door
        :return:
        """
        double_door_world = door._world
        door_hinge_kse = door.body.parent_kinematic_structure_entity
        double_door_C_door: RevoluteConnection = door_hinge_kse.parent_connection
        double_door_T_door = double_door_C_door.parent_T_connection_expression
        parent_T_door = parent_T_double_door @ double_door_T_door
        old_dof = double_door_C_door.dof
        door_world = double_door_world.move_branch_to_new_world(door_hinge_kse)

        new_dof = DegreeOfFreedom(
            name=old_dof.name,
            lower_limits=old_dof.lower_limits,
            upper_limits=old_dof.upper_limits,
        )

        new_parent_C_left = RevoluteConnection(
            parent=new_parent,
            child=door_hinge_kse,
            parent_T_connection_expression=parent_T_door,
            multiplier=double_door_C_door.multiplier,
            offset=double_door_C_door.offset,
            axis=double_door_C_door.axis,
            dof_id=new_dof.id,
        )

        with double_door_world.modify_world(), door_world.modify_world():
            double_door_world.remove_semantic_annotation(door)
            door_world.add_semantic_annotation(door)

        return door_world, new_parent_C_left, new_dof

    def remove_doors_from_world(
        self, parent_world: World, wall_event_thickness: float = 0.1
    ):
        """
        Remove the door volumes from all bodies in the world that are not doors.

        :param parent_world: The world from which to remove the door volumes.
        :param wall_event_thickness: The thickness of the wall event used to create the door events.
        """
        doors: List[Door] = parent_world.get_semantic_annotations_by_type(Door)
        if not doors:
            return
        all_doors_event = self._build_all_doors_event_from_semantic_annotations(
            doors, wall_event_thickness
        )

        all_bodies_not_door = self._get_all_bodies_excluding_doors_from_world(
            parent_world
        )

        if not all_doors_event.is_empty():
            self._remove_doors_from_bodies(all_bodies_not_door, all_doors_event)

    @staticmethod
    def _get_all_bodies_excluding_doors_from_world(world: World) -> List[Body]:
        """
        Return all bodies in the world that are not part of any door semantic annotation.

        :param world: The world from which to get the bodies.
        :return: A list of bodies that are not part of any door semantic annotation.
        """
        all_doors = let(Door, domain=world.semantic_annotations)
        other_body = let(type_=Body, domain=world.bodies)
        door_bodies = all_doors.bodies
        bodies_without_excluded_bodies_query = an(
            entity(other_body, for_all(door_bodies, not_(in_(other_body, door_bodies))))
        )

        filtered_bodies = list(bodies_without_excluded_bodies_query.evaluate())
        return filtered_bodies

    def _build_all_doors_event_from_semantic_annotations(
        self, doors: List[Door], wall_event_thickness: float = 0.1
    ) -> Event:
        """
        Build a single event representing all doors by combining the events of each door.

        :param doors: The list of door semantic annotations to build the event from.
        :param wall_event_thickness: The thickness of the wall event used to create the door events.
        :return: An event representing all doors.
        """
        door_events = [
            self._build_single_door_event(door, wall_event_thickness) for door in doors
        ]
        if door_events:
            return reduce(or_, door_events)
        return Event()

    @staticmethod
    def _build_single_door_event(
        door: Door, wall_event_thickness: float = 0.1
    ) -> Event:
        """
        Build an event representing a single door by creating a bounding box event around the door's collision shapes

        :param door: The door semantic annotation to build the event from.
        :param wall_event_thickness: The thickness of the wall event used to create the door event.
        :return: An event representing the door.
        """
        door_event = door.body.collision.as_bounding_box_collection_in_frame(
            door._world.root
        ).event

        door_plane_spatial_variables = SpatialVariables.yz
        door_thickness_spatial_variable = SpatialVariables.x.value
        door_event = door_event.marginal(door_plane_spatial_variables)
        door_event.fill_missing_variables([door_thickness_spatial_variable])
        thickness_event = SimpleEvent(
            {
                door_thickness_spatial_variable: closed(
                    -wall_event_thickness / 2, wall_event_thickness / 2
                )
            }
        ).as_composite_set()
        thickness_event.fill_missing_variables(door_plane_spatial_variables)
        door_event = door_event & thickness_event

        return door_event

    def _remove_doors_from_bodies(self, bodies: List[Body], all_doors_event: Event):
        """
        Remove the door volumes from the given bodies by subtracting the all_doors_event from each body's collision event.

        :param bodies: The list of bodies from which to remove the door volumes.
        :param all_doors_event: The event representing all doors.
        """
        for body in bodies:
            self._remove_door_from_body(body, all_doors_event)

    @staticmethod
    def _remove_door_from_body(body: Body, all_doors_event: Event):
        """
        Remove the door volumes from the given body by subtracting the all_doors_event from the body's collision event.

        :param body: The body from which to remove the door volumes.
        :param all_doors_event: The event representing all doors.
        """
        root = body._world.root
        body_event = (
            body.collision.as_bounding_box_collection_in_frame(root).event
            - all_doors_event
        )
        new_collision = BoundingBoxCollection.from_event(root, body_event).as_shapes()
        body.collision = new_collision
        body.visual = new_collision


@dataclass
class HasDrawerFactories(ABC):
    """
    Mixin for factories receiving multiple DrawerFactories.
    """

    drawer_factory_configs: List[DrawerConfigForParentFactory]

    def add_drawers_to_world(self, parent_world: World):
        """
        Adds drawers to the parent world. A prismatic connection is created for each drawer.
        """

        for config in self.drawer_factory_configs:
            self._add_drawer_to_world(
                drawer_factory=config.factory_instance,
                parent_T_drawer=config.parent_T_child,
                parent_world=parent_world,
            )

    @staticmethod
    def _create_drawer_upper_lower_limits(
        drawer_factory: DrawerFactory,
    ) -> Tuple[DerivativeMap[float], DerivativeMap[float]]:
        """
        Return the upper and lower limits for the drawer's degree of freedom.
        """
        lower_limits = DerivativeMap[float]()
        upper_limits = DerivativeMap[float]()
        lower_limits.position = 0.0
        upper_limits.position = (
            drawer_factory.container_factory_config.factory_instance.scale.x * 0.75
        )

        return upper_limits, lower_limits


@dataclass
class ContainerConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating a container from a parent factory.
    """

    factory_instance: ContainerFactory


@dataclass
class ContainerFactory(
    SemanticAnnotationFactory[Corpus, ContainerConfigForParentFactory]
):
    """
    Factory for creating a container with walls of a specified thickness and its opening in direction.
    """

    scale: Scale = field(default_factory=lambda: Scale(1.0, 1.0, 1.0))
    """
    The scale of the container, defining its size in the world.
    """

    wall_thickness: float = 0.05
    """
    The thickness of the walls of the container.
    """

    direction: Direction = field(default=Direction.X)
    """
    The direction in which the container is open.
    """

    @property
    def _config_type_for_parent_factory(self) -> Type[ContainerConfigForParentFactory]:
        return ContainerConfigForParentFactory

    def get_config_for_parent_factory(
        self,
        parent_T_child: Optional[TransformationMatrix],
        *args,
        **kwargs,
    ) -> ContainerConfigForParentFactory:
        """
        Return the configuration for the parent factory.
        """
        return self._config_type_for_parent_factory(
            factory_instance=self, parent_T_child=parent_T_child
        )

    def _create(self, world: World) -> World:
        """
        Return a world with a container body at its root.
        """


@dataclass
class HandleConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating a handle from a parent factory.
    """

    factory_instance: HandleFactory

    semantic_handle_position: Optional[SemanticPositionDescription] = field(
        default=None
    )

    def __post_init__(self):
        assert (
            self.parent_T_child is not None or self.semantic_handle_position is not None
        ), "Either parent_T_handle or semantic_position must be set."
        if (
            self.parent_T_child is not None
            and self.semantic_handle_position is not None
        ):
            logging.warning(
                f"During the creation of a factory, both parent_T_handle and semantic_position were set. Prioritizing parent_T_handle."
            )


@dataclass
class HandleFactory(SemanticAnnotationFactory[Handle, HandleConfigForParentFactory]):
    """
    Factory for creating a handle with a specified scale and thickness.
    The handle is represented as a box with an inner cutout to create the handle shape.
    """

    scale: Scale = field(default_factory=lambda: Scale(0.05, 0.1, 0.02))
    """
    The scale of the handle.
    """

    thickness: float = 0.01
    """
    Thickness of the handle bar.
    """

    @property
    def _config_type_for_parent_factory(self) -> Type[HandleConfigForParentFactory]:
        return HandleConfigForParentFactory

    def get_config_for_parent_factory(
        self,
        parent_T_child: Optional[TransformationMatrix] = None,
        semantic_handle_position: Optional[SemanticPositionDescription] = None,
        *args,
        **kwargs,
    ) -> HandleConfigForParentFactory:
        """
        Return the configuration for the parent factory.
        """
        return self._config_type_for_parent_factory(
            factory_instance=self,
            parent_T_child=parent_T_child,
            semantic_handle_position=semantic_handle_position,
        )

    def _create(self, world: World) -> World:
        """
        Create a world with a handle body at its root.
        """

        handle_event = self._create_handle_event()

        handle = Body(name=self.name)
        collision = BoundingBoxCollection.from_event(handle, handle_event).as_shapes()
        handle.collision = collision
        handle.visual = collision

        semantic_handle_annotation = Handle(name=self.name, body=handle)

        world.add_kinematic_structure_entity(handle)
        world.add_semantic_annotation(semantic_handle_annotation)
        return world

    def _create_handle_event(self) -> Event:
        """
        Return an event representing a handle.
        """

        handle_event = self._create_outer_box_event().as_composite_set()

        inner_box = self._create_inner_box_event().as_composite_set()

        handle_event -= inner_box

        return handle_event

    def _create_outer_box_event(self) -> SimpleEvent:
        """
        Return an event representing the main body of a handle.
        """
        x_interval = closed(0, self.scale.x)
        y_interval = closed(-self.scale.y / 2, self.scale.y / 2)
        z_interval = closed(-self.scale.z / 2, self.scale.z / 2)

        handle_event = SimpleEvent(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )

        return handle_event

    def _create_inner_box_event(self) -> SimpleEvent:
        """
        Return an event used to cut out the inner part of the handle.
        """
        x_interval = closed(0, self.scale.x - self.thickness)
        y_interval = closed(
            -self.scale.y / 2 + self.thickness, self.scale.y / 2 - self.thickness
        )
        z_interval = closed(-self.scale.z, self.scale.z)

        inner_box = SimpleEvent(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )

        return inner_box


@dataclass
class DoorLikeFactory(SemanticAnnotationFactory[T, GenericConfigForParentFactory], ABC):
    """
    Abstract factory for creating door-like factories such as doors or double doors.
    """


@dataclass
class DoorLikeConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating door-like factories from a parent factory.
    """


@dataclass
class DoorConfigForParentFactory(DoorLikeConfigForParentFactory):
    """
    Configuration for creating a door from a parent factory.
    """

    factory_instance: DoorFactory

    hinge_axis: Vector3


@dataclass
class DoorFactory(DoorLikeFactory[Door, DoorConfigForParentFactory], HasHandleFactory):
    """
    Factory for creating a door with a handle. The door is defined by its scale and handle direction.
    The doors origin is at the pivot point of the door, not at the center.
    """

    scale: Scale = field(default_factory=lambda: Scale(0.03, 1.0, 2.0))
    """
    The scale of the entryway.
    """

    @property
    def _config_type_for_parent_factory(self) -> Type[DoorConfigForParentFactory]:
        return DoorConfigForParentFactory

    def get_config_for_parent_factory(
        self,
        parent_T_child: Optional[TransformationMatrix],
        hinge_axis: Vector3,
        *args,
        **kwargs,
    ) -> DoorConfigForParentFactory:
        """
        Return the configuration for the parent factory.
        """
        return self._config_type_for_parent_factory(
            factory_instance=self, parent_T_child=parent_T_child, hinge_axis=hinge_axis
        )

    def _create(self, world: World) -> World:
        """
        Return a world with a door body at its root. The door has a handle and is defined by its scale and handle direction.
        """

        door_event = self.scale.simple_event.as_composite_set()

        body = Body(name=self.name)
        bounding_box_collection = BoundingBoxCollection.from_event(body, door_event)
        collision = bounding_box_collection.as_shapes()
        body.collision = collision
        body.visual = collision

        world.add_kinematic_structure_entity(body)

        self.create_parent_T_handle_from_parent_scale(self.scale)

        door_T_handle = (
            self.handle_factory_config.parent_T_child
            or self.create_parent_T_handle_from_parent_scale(self.scale)
        )
        self.add_handle_to_world(door_T_handle, world)
        semantic_handle_annotation: Handle = world.get_semantic_annotations_by_type(
            Handle
        )[0]
        world.add_semantic_annotation(
            Door(name=self.name, handle=semantic_handle_annotation, body=body)
        )

        return world


@dataclass
class DoubleDoorConfigForParentFactory(DoorLikeConfigForParentFactory):
    """
    Configuration for creating a double door from a parent factory.
    """

    factory_instance: DoubleDoorFactory


@dataclass
class DoubleDoorFactory(
    DoorLikeFactory[DoubleDoor, DoubleDoorConfigForParentFactory], HasDoorLikeFactories
):
    """
    Factory for creating a double door with two doors and their handles.
    """

    @property
    def _config_type_for_parent_factory(self) -> Type[DoubleDoorConfigForParentFactory]:
        return DoubleDoorConfigForParentFactory

    def get_config_for_parent_factory(
        self,
        parent_T_child: Optional[TransformationMatrix],
        *args,
        **kwargs,
    ) -> DoubleDoorConfigForParentFactory:
        """
        Return the configuration for the parent factory.
        """
        return self._config_type_for_parent_factory(
            factory_instance=self, parent_T_child=parent_T_child
        )

    def _create(self, world: World) -> World:
        """
        Return a world with a virtual body at its root that is the parent of the two doors making up the double door.
        """

        double_door_body = Body(name=self.name)
        world.add_kinematic_structure_entity(double_door_body)

        self.add_doorlike_semantic_annotation_to_world(
            parent_world=world,
        )

        semantic_door_annotations = world.get_semantic_annotations_by_type(Door)
        assert (
            len(semantic_door_annotations) == 2
        ), "Double door must have exactly two doors semantic annotations"

        left_door, right_door = semantic_door_annotations
        if (
            left_door.body.parent_connection.origin_expression.y
            > right_door.body.parent_connection.origin_expression.y
        ):
            right_door, left_door = (
                semantic_door_annotations[0],
                semantic_door_annotations[1],
            )

        semantic_double_door_annotation = DoubleDoor(
            left_door=left_door, right_door=right_door
        )
        world.add_semantic_annotation(semantic_double_door_annotation)
        return world


@dataclass
class DrawerConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating a drawer from a parent factory.
    """

    factory_instance: DrawerFactory


@dataclass
class DrawerFactory(
    SemanticAnnotationFactory[Drawer, DrawerConfigForParentFactory], HasHandleFactory
):
    """
    Factory for creating a drawer with a handle and a container.
    """

    container_factory_config: ContainerConfigForParentFactory
    """
    The factory used to create the container of the drawer.
    """

    @property
    def _config_type_for_parent_factory(self) -> Type[DrawerConfigForParentFactory]:
        return DrawerConfigForParentFactory

    def get_config_for_parent_factory(
        self,
        parent_T_child: Optional[TransformationMatrix],
        *args,
        **kwargs,
    ) -> DrawerConfigForParentFactory:
        """
        Return the configuration for the parent factory.
        """
        return self._config_type_for_parent_factory(
            factory_instance=self, parent_T_child=parent_T_child
        )

    def _create(self, world: World) -> World:
        """
        Return a world with a drawer at its root. The drawer consists of a container and a handle.
        """

        container_world = self.container_factory_config.factory_instance.create()
        parent_T_handle = (
            self.handle_factory_config.parent_T_child
            or self.create_parent_T_handle_from_parent_scale(
                self.container_factory_config.factory_instance.scale
            )
        )

        self.add_handle_to_world(parent_T_handle, container_world)

        semantic_container_annotation: Corpus = (
            container_world.get_semantic_annotations_by_type(Corpus)[0]
        )
        semantic_handle_annotation: Handle = (
            container_world.get_semantic_annotations_by_type(Handle)[0]
        )
        semantic_drawer_annotation = Drawer(
            name=self.name,
            container=semantic_container_annotation,
            handle=semantic_handle_annotation,
        )
        with container_world.modify_world():
            container_world.add_semantic_annotation(semantic_drawer_annotation)
        container_world.name = world.name
        return container_world


@dataclass
class DresserConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating a dresser from a parent factory.
    """

    factory_instance: DresserFactory


@dataclass
class DresserFactory(
    SemanticAnnotationFactory[Dresser, DresserConfigForParentFactory],
    HasDoorLikeFactories,
    HasDrawerFactories,
):
    """
    Factory for creating a dresser with drawers, and doors.
    """

    def _create(self, world: World) -> World:
        """
        Return a world with a dresser at its root. The dresser consists of a container, potentially drawers, and doors.
        Assumes that the number of drawers matches the number of drawer transforms.
        """

        dresser_world = self._make_dresser_world()
        dresser_world.name = world.name
        return self._make_interior(dresser_world)

    def _make_dresser_world(self) -> World:
        """
        Create a world with a dresser semantic annotation that contains a container, drawers, and doors, but no interior yet.
        """
        dresser_world = self.container_factory_config.factory_instance.create()
        with dresser_world.modify_world():
            semantic_container_annotation: Corpus = (
                dresser_world.get_semantic_annotations_by_type(Corpus)[0]
            )

            self.add_doorlike_semantic_annotation_to_world(dresser_world)

            self.add_drawers_to_world(dresser_world)

            semantic_dresser_annotation = Dresser(
                name=self.name,
                container=semantic_container_annotation,
                drawers=dresser_world.get_semantic_annotations_by_type(Drawer),
                doors=dresser_world.get_semantic_annotations_by_type(Door),
            )
            dresser_world.add_semantic_annotation(semantic_dresser_annotation)
            dresser_world.name = self.name.name

        return dresser_world

    def _make_interior(self, world: World) -> World:
        """
        Create the interior of the dresser by subtracting the drawers and doors from the container, and filling  with
        the remaining space.

        :param world: The world containing the dresser body as its root.
        """
        dresser_body: Body = world.root
        container_event = dresser_body.collision.as_bounding_box_collection_at_origin(
            TransformationMatrix(reference_frame=dresser_body)
        ).event

        container_footprint = self._subtract_bodies_from_container_footprint(
            world, container_event
        )

        container_event = self._fill_container_body(
            container_footprint, container_event
        )

        collision_shapes = BoundingBoxCollection.from_event(
            dresser_body, container_event
        ).as_shapes()
        dresser_body.collision = collision_shapes
        dresser_body.visual = collision_shapes
        return world

    def _subtract_bodies_from_container_footprint(
        self, world: World, container_event: Event
    ) -> Event:
        """
        Subtract the bounding boxes of all bodies in the world from the container event,
        except for the dresser body itself. This creates a frontal footprint of the container

        :param world: The world containing the dresser body as its root.
        :param container_event: The event representing the container.

        :return: An event representing the footprint of the container after subtracting other bodies.
        """
        dresser_body = world.root

        container_footprint = container_event.marginal(SpatialVariables.yz)

        for body in world.bodies_with_enabled_collision:
            if body == dresser_body:
                continue
            body_footprint = body.collision.as_bounding_box_collection_at_origin(
                TransformationMatrix(reference_frame=dresser_body)
            ).event.marginal(SpatialVariables.yz)
            container_footprint -= body_footprint
            if container_footprint.is_empty():
                return Event()

        return container_footprint

    def _fill_container_body(
        self, container_footprint: Event, container_event: Event
    ) -> Event:
        """
        Expand container footprint into 3d space and fill the space of the resulting container body.

        :param container_footprint: The footprint of the container in the yz-plane.
        :param container_event: The event representing the container.

        :return: An event representing the container body with the footprint filled in the x-direction.
        """

        container_footprint.fill_missing_variables([SpatialVariables.x.value])

        depth_interval = container_event.bounding_box()[SpatialVariables.x.value]
        limiting_event = SimpleEvent(
            {SpatialVariables.x.value: depth_interval}
        ).as_composite_set()
        limiting_event.fill_missing_variables(SpatialVariables.yz)

        container_event |= container_footprint & limiting_event

        return container_event


@dataclass
class RoomConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating a room from a parent factory.
    """

    factory_instance: RoomFactory


@dataclass
class RoomFactory(SemanticAnnotationFactory[Room, RoomConfigForParentFactory]):
    """
    Factory for creating a room with a specific region.
    """

    floor_polytope: List[Point3]
    """
    The region that defines the room's boundaries and reference frame.
    """

    @property
    def _config_type_for_parent_factory(self) -> Type[RoomConfigForParentFactory]:
        return RoomConfigForParentFactory

    def get_config_for_parent_factory(
        self,
        parent_T_child: Optional[TransformationMatrix],
        *args,
        **kwargs,
    ) -> RoomConfigForParentFactory:
        """
        Return the configuration for the parent factory.
        """
        return self._config_type_for_parent_factory(
            factory_instance=self, parent_T_child=parent_T_child
        )

    def _create(self, world: World) -> World:
        """
        Return a world with a room semantic annotation that contains the specified region.
        """
        room_body = Body(name=self.name)
        world.add_kinematic_structure_entity(room_body)

        region = Region.from_3d_points(
            points_3d=self.floor_polytope,
            name=PrefixedName(self.name.name + "_surface_region", self.name.prefix),
            reference_frame=room_body,
        )
        connection = FixedConnection(
            parent=room_body,
            child=region,
            parent_T_connection_expression=TransformationMatrix(),
        )
        world.add_connection(connection)

        floor = Floor(
            name=PrefixedName(self.name.name + "_floor", self.name.prefix),
            supporting_surface=region,
        )
        world.add_semantic_annotation(floor)
        semantic_room_annotation = Room(name=self.name, floor=floor)
        world.add_semantic_annotation(semantic_room_annotation)

        return world


@dataclass
class WallConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating a wall from a parent factory.
    """

    factory_instance: WallFactory


@dataclass
class WallFactory(
    SemanticAnnotationFactory[Wall, WallConfigForParentFactory], HasDoorLikeFactories
):

    scale: Scale = field(kw_only=True)
    """
    The scale of the wall.
    """

    @property
    def _config_type_for_parent_factory(self) -> Type[WallConfigForParentFactory]:
        return WallConfigForParentFactory

    def get_config_for_parent_factory(
        self,
        parent_T_child: Optional[TransformationMatrix],
        *args,
        **kwargs,
    ) -> WallConfigForParentFactory:
        """
        Return the configuration for the parent factory.
        """
        return self._config_type_for_parent_factory(
            factory_instance=self, parent_T_child=parent_T_child
        )

    def _create(self, world: World) -> World:
        """
        Return a world with the wall body at its root and potentially doors and double doors as children of the wall body.
        """
        wall_world = self._create_wall_world()
        self.add_doorlike_semantic_annotation_to_world(wall_world)
        self.remove_doors_from_world(wall_world)
        world.merge_world(wall_world)

        return world

    def _create_wall_world(self) -> World:
        wall_world = World()
        with wall_world.modify_world():
            wall_body = Body(name=self.name)
            wall_collision = self._create_wall_collision(wall_body)
            wall_body.collision = wall_collision
            wall_body.visual = wall_collision
            with wall_world.modify_world():
                wall_world.add_kinematic_structure_entity(wall_body)

            wall = Wall(
                name=self.name,
                body=wall_body,
            )

            wall_world.add_semantic_annotation(wall)

        return wall_world

    def _create_wall_collision(self, reference_frame: Body) -> ShapeCollection:
        """
        Return the collision shapes for the wall. A wall event is created based on the scale of the wall, and
        doors are removed from the wall event. The resulting bounding box collection is converted to shapes.
        """

        wall_event = self._create_wall_event().as_composite_set()

        bounding_box_collection = BoundingBoxCollection.from_event(
            reference_frame, wall_event
        )

        wall_collision = bounding_box_collection.as_shapes()
        return wall_collision

    def _create_wall_event(self) -> SimpleEvent:
        """
        Return a wall event created from its scale. The height origin is on the ground, not in the center of the wall.
        """
        x_interval = closed(-self.scale.x / 2, self.scale.x / 2)
        y_interval = closed(-self.scale.y / 2, self.scale.y / 2)
        z_interval = closed(0, self.scale.z)

        wall_event = SimpleEvent(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )
        return wall_event
