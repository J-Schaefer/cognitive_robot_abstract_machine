import unittest

import pytest
import rclpy

from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import InvalidDoorDimensions
from semantic_digital_twin.semantic_annotations.mixins import HasCase

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Drawer,
    Dresser,
    Wall,
    Hinge,
    DoubleDoor,
    Fridge,
    Slider,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    TransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


class TestFactories(unittest.TestCase):
    def test_handle_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        returned_handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"),
            scale=Scale(0.1, 0.2, 0.03),
            thickness=0.03,
            world=world,
            parent=root,
        )
        semantic_handle_annotations = world.get_semantic_annotations_by_type(Handle)
        self.assertEqual(len(semantic_handle_annotations), 1)

        queried_handle: Handle = semantic_handle_annotations[0]
        self.assertEqual(returned_handle, queried_handle)
        self.assertEqual(
            world.root, queried_handle.body.parent_kinematic_structure_entity
        )

    def test_basic_has_body_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        returned_hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"),
            world=world,
            parent=root,
        )
        returned_slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"),
            world=world,
            parent=root,
        )
        semantic_hinge_annotations = world.get_semantic_annotations_by_type(Hinge)
        self.assertEqual(len(semantic_hinge_annotations), 1)

        queried_hinge: Hinge = semantic_hinge_annotations[0]
        self.assertEqual(returned_hinge, queried_hinge)
        self.assertEqual(
            world.root, queried_hinge.body.parent_kinematic_structure_entity
        )
        semantic_slider_annotations = world.get_semantic_annotations_by_type(Slider)
        self.assertEqual(len(semantic_slider_annotations), 1)
        queried_slider: Slider = semantic_slider_annotations[0]
        self.assertEqual(returned_slider, queried_slider)
        self.assertEqual(
            world.root, queried_slider.body.parent_kinematic_structure_entity
        )

    def test_door_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        returned_door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world, parent=root
        )
        semantic_door_annotations = world.get_semantic_annotations_by_type(Door)
        self.assertEqual(len(semantic_door_annotations), 1)

        queried_door: Door = semantic_door_annotations[0]
        self.assertEqual(returned_door, queried_door)
        self.assertEqual(
            world.root, queried_door.body.parent_kinematic_structure_entity
        )

    def test_door_factory_invalid(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with pytest.raises(InvalidDoorDimensions):
            Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                scale=Scale(1, 1, 2),
                world=world,
                parent=root,
            )

        with pytest.raises(InvalidDoorDimensions):
            Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                scale=Scale(1, 2, 1),
                world=world,
                parent=root,
            )

    def test_has_hinge_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world, parent=root
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, parent=root
        )
        assert len(world.kinematic_structure_entities) == 3

        door.add_hinge(hinge)

        assert door.body.parent_kinematic_structure_entity == hinge.body
        assert isinstance(hinge.body.parent_connection, RevoluteConnection)
        assert door.hinge == hinge

    def test_reverse_has_hinge_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world, parent=root
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, parent=door.body
        )
        assert len(world.kinematic_structure_entities) == 3

        door.add_hinge(hinge)

        assert door.body.parent_kinematic_structure_entity == hinge.body
        assert isinstance(hinge.body.parent_connection, RevoluteConnection)
        assert door.hinge == hinge

    def test_has_handle_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)

        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"),
            scale=Scale(0.03, 1, 2),
            world=world,
            parent=root,
        )

        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"),
            world=world,
            parent=root,
        )
        assert len(world.kinematic_structure_entities) == 3

        assert root == handle.body.parent_kinematic_structure_entity

        door.add_handle(handle)

        assert door.body == handle.body.parent_kinematic_structure_entity
        assert door.handle == handle

    def test_double_door_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        left_door = Door.create_with_new_body_in_world(
            name=PrefixedName("left_door"), world=world, parent=root
        )
        right_door = Door.create_with_new_body_in_world(
            name=PrefixedName("right_door"), world=world, parent=root
        )
        double_door = DoubleDoor.create_with_left_right_door_in_world(
            left_door, right_door
        )
        semantic_double_door_annotations = world.get_semantic_annotations_by_type(
            DoubleDoor
        )

        self.assertEqual(len(semantic_double_door_annotations), 1)

    def test_case_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("case"),
            world=world,
            parent=root,
            scale=Scale(1, 1, 2.0),
        )

        assert isinstance(fridge, HasCase)

        semantic_container_annotations = world.get_semantic_annotations_by_type(Fridge)
        self.assertEqual(len(semantic_container_annotations), 1)

        assert len(world.get_semantic_annotations_by_type(HasCase)) == 1

    def test_drawer_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"),
            world=world,
            parent=root,
            scale=Scale(0.2, 0.3, 0.2),
        )
        assert isinstance(drawer, HasCase)
        semantic_drawer_annotations = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(len(semantic_drawer_annotations), 1)

    def test_has_slider_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"),
            world=world,
            parent=root,
            scale=Scale(0.2, 0.3, 0.2),
        )
        slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"), world=world, parent=root
        )
        assert len(world.kinematic_structure_entities) == 3

        drawer.add_slider(slider)

        assert drawer.body.parent_kinematic_structure_entity == slider.body
        assert isinstance(slider.body.parent_connection, PrismaticConnection)
        assert drawer.slider == slider

    def test_reverse_has_slider_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"),
            scale=Scale(0.2, 0.3, 0.2),
            world=world,
            parent=root,
        )
        slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"), world=world, parent=drawer.body
        )
        assert len(world.kinematic_structure_entities) == 3

        drawer.add_slider(slider)

        assert drawer.body.parent_kinematic_structure_entity == slider.body
        assert isinstance(slider.body.parent_connection, PrismaticConnection)
        assert drawer.slider == slider

    # def test_wall_factory(self):
    #     handle_factory = HandleFactory(name=PrefixedName("handle"))
    #     handle_factory_config = handle_factory.get_config_for_parent_factory(
    #         semantic_handle_position=SemanticPositionDescription(
    #             horizontal_direction_chain=[
    #                 HorizontalSemanticDirection.RIGHT,
    #                 HorizontalSemanticDirection.FULLY_CENTER,
    #             ],
    #             vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
    #         ),
    #     )
    #     door_factory = DoorFactory(
    #         name=PrefixedName("door"),
    #         handle_factory_config=handle_factory_config,
    #     )
    #     door_factory_config = door_factory.get_config_for_parent_factory(
    #         parent_T_child=TransformationMatrix.from_xyz_rpy(y=-0.5),
    #         hinge_axis=Vector3.Z(),
    #     )
    #
    #     handle_factory2 = HandleFactory(name=PrefixedName("handle2"))
    #     handle_factory_config2 = handle_factory.get_config_for_parent_factory(
    #         semantic_handle_position=SemanticPositionDescription(
    #             horizontal_direction_chain=[
    #                 HorizontalSemanticDirection.LEFT,
    #                 HorizontalSemanticDirection.FULLY_CENTER,
    #             ],
    #             vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
    #         ),
    #     )
    #     door_factory2 = DoorFactory(
    #         name=PrefixedName("door2"),
    #         handle_factory_config=handle_factory_config2,
    #     )
    #     door_factory_config2 = door_factory2.get_config_for_parent_factory(
    #         parent_T_child=TransformationMatrix.from_xyz_rpy(y=0.5),
    #         hinge_axis=Vector3.Z(),
    #     )
    #
    #     double_door_factory = DoubleDoorFactory(
    #         name=PrefixedName("double_door"),
    #         door_like_factory_configs=[door_factory_config, door_factory_config2],
    #     )
    #     double_door_factory_config = double_door_factory.get_config_for_parent_factory(
    #         parent_T_child=TransformationMatrix.from_xyz_rpy(x=0.5)
    #     )
    #
    #     single_door_handle_factory = HandleFactory(
    #         name=PrefixedName("single_door_handle")
    #     )
    #     single_door_handle_factory_config = (
    #         single_door_handle_factory.get_config_for_parent_factory(
    #             semantic_handle_position=SemanticPositionDescription(
    #                 horizontal_direction_chain=[
    #                     HorizontalSemanticDirection.RIGHT,
    #                     HorizontalSemanticDirection.FULLY_CENTER,
    #                 ],
    #                 vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
    #             ),
    #         )
    #     )
    #     single_door_factory = DoorFactory(
    #         name=PrefixedName("single_door"),
    #         handle_factory_config=single_door_handle_factory_config,
    #     )
    #     single_door_factory_config = (
    #         single_door_handle_factory.get_config_for_parent_factory(
    #             parent_T_child=TransformationMatrix.from_xyz_rpy(y=-1.5),
    #             hinge_axis=Vector3.Z(),
    #         )
    #     )
    #
    #     factory = WallFactory(
    #         name=PrefixedName("wall"),
    #         scale=Scale(0.1, 4, 2),
    #         door_like_factory_configs=[
    #             double_door_factory_config,
    #             single_door_factory_config,
    #         ],
    #     )
    #     world = factory.create()
    #     semantic_wall_annotations = world.get_semantic_annotations_by_type(Wall)
    #     self.assertEqual(len(semantic_wall_annotations), 1)
    #
    #     wall: Wall = semantic_wall_annotations[0]
    #     self.assertEqual(world.root, wall.body)


if __name__ == "__main__":
    unittest.main()
