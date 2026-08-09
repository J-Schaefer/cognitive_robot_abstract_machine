from __future__ import annotations

import time

import numpy as np
import pytest

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.simulated_cable import SimulatedCable
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World


class TestSimulatedCableInMuJoCo:
    """
    Integration tests for the cable chain in MuJoCo simulation.
    """

    STEP_SIZE = 0.001

    @staticmethod
    def _create_world_with_cable(
        world: World,
        number_of_links: int = 5,
        length: float = 0.3,
        thickness: float = 0.005,
    ) -> tuple[SimulatedCable, Body]:
        floor = Body(name=PrefixedName("floor"))
        floor.collision = ShapeCollection(
            [
                Box(
                    scale=Scale(2.0, 2.0, 0.05),
                    color=Color(1.0, 1.0, 0.5, 1.0),
                )
            ],
            reference_frame=floor,
        )
        anchor = Body(name=PrefixedName("cable_anchor"))
        anchor.collision = ShapeCollection(
            [
                Box(
                    scale=Scale(0.05, 0.05, 0.05),
                    color=Color(1.0, 0.0, 0.0, 1.0),
                )
            ],
            reference_frame=anchor,
        )
        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=floor))
            world.add_connection(FixedConnection(parent=floor, child=anchor))
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=anchor,
                length=length,
                thickness=thickness,
                number_of_links=number_of_links,
            )
        return cable, anchor

    @pytest.fixture
    def world_with_cable(self):
        world = World()
        multi_sim = MujocoSim(world=world, headless=True, step_size=self.STEP_SIZE)
        try:
            multi_sim.start_simulation()
            time.sleep(0.5)
            cable, anchor = self._create_world_with_cable(
                world, number_of_links=5, length=0.25, thickness=0.005
            )
            yield world, cable, anchor, multi_sim
        finally:
            multi_sim.stop_simulation()

    def test_cable_links_have_expected_names_in_sim(self, world_with_cable):
        """
        Cable link bodies are created with expected names in the simulator.
        """
        world, cable, anchor, multi_sim = world_with_cable
        time.sleep(0.5)
        body_names = multi_sim.simulator.callbacks["get_all_body_names"]().result
        assert "cable_link_0" in body_names
        assert "cable_link_1" in body_names
        assert "cable_link_2" in body_names

    def test_cable_hangs_below_anchor(self, world_with_cable):
        """
        After settling, cable links are below the anchor.
        """
        world, cable, anchor, multi_sim = world_with_cable
        time.sleep(1.0)

        anchor_pos = multi_sim.simulator.callbacks["get_body_position"](
            body_name="cable_anchor"
        ).result
        link_pos = multi_sim.simulator.callbacks["get_body_position"](
            body_name="cable_link_4"
        ).result
        assert link_pos[2] < anchor_pos[2]

    def test_can_weld_and_unweld_cable_link(self, world_with_cable):
        """
        A cable link can be welded to another body and subsequently unwelded.
        """
        world, cable, anchor, multi_sim = world_with_cable
        time.sleep(0.5)

        result = multi_sim.simulator.callbacks["weld_bodies"](
            body_1_name="cable_link_2",
            body_2_name="cable_anchor",
        )
        assert result.type.value == 1

        time.sleep(0.3)

        result = multi_sim.simulator.callbacks["unweld_bodies"](
            body_1_name="cable_link_2",
            body_2_name="cable_anchor",
        )
        assert result.type.value == 1

    def test_cable_grasp_does_not_break_chain(self, world_with_cable):
        """
        Welding a link to the anchor keeps the ball joint chain intact.
        """
        world, cable, anchor, multi_sim = world_with_cable
        time.sleep(0.5)

        multi_sim.simulator.callbacks["weld_bodies"](
            body_1_name="cable_link_2",
            body_2_name="cable_anchor",
        )
        time.sleep(0.5)

        link0_pos = multi_sim.simulator.callbacks["get_body_position"](
            body_name="cable_link_0"
        ).result
        link1_pos = multi_sim.simulator.callbacks["get_body_position"](
            body_name="cable_link_1"
        ).result
        link2_pos = multi_sim.simulator.callbacks["get_body_position"](
            body_name="cable_link_2"
        ).result
        link3_pos = multi_sim.simulator.callbacks["get_body_position"](
            body_name="cable_link_3"
        ).result
        link4_pos = multi_sim.simulator.callbacks["get_body_position"](
            body_name="cable_link_4"
        ).result

        d01 = np.linalg.norm(link0_pos[:3] - link1_pos[:3])
        d12 = np.linalg.norm(link1_pos[:3] - link2_pos[:3])
        d23 = np.linalg.norm(link2_pos[:3] - link3_pos[:3])
        d34 = np.linalg.norm(link3_pos[:3] - link4_pos[:3])
        assert d01 < 0.1, "link 0-1 disconnected"
        assert d12 < 0.1, "link 1-2 disconnected"
        assert d23 < 0.1, "link 2-3 disconnected"
        assert d34 < 0.1, "link 3-4 disconnected"


class TestCableReparentInSimulation:
    """
    Tests that the cable chain survives full kinematic reparenting during
    re-attachment in MuJoCo.
    """

    STEP_SIZE = 0.001

    @staticmethod
    def _create_world_with_full_setup(world: World) -> tuple[SimulatedCable, Body, Body]:
        floor = Body(name=PrefixedName("floor"))
        floor.collision = ShapeCollection(
            [Box(scale=Scale(2.0, 2.0, 0.05), color=Color(1.0, 1.0, 0.5, 1.0))],
            reference_frame=floor,
        )
        anchor = Body(name=PrefixedName("cable_anchor"))
        anchor.collision = ShapeCollection(
            [Box(scale=Scale(0.05, 0.05, 0.05), color=Color(1.0, 0.0, 0.0, 1.0))],
            reference_frame=anchor,
        )
        gripper = Body(name=PrefixedName("gripper"))
        gripper.collision = ShapeCollection(
            [Box(scale=Scale(0.03, 0.03, 0.03), color=Color(0.0, 0.0, 1.0, 1.0))],
            reference_frame=gripper,
        )
        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=floor))
            world.add_connection(FixedConnection(parent=floor, child=anchor))
            world.add_connection(FixedConnection(parent=floor, child=gripper))
            cable = SimulatedCable.create_in_world(
                name=PrefixedName("cable"),
                world=world,
                hanging_from=anchor,
                length=0.25,
                thickness=0.005,
                number_of_links=5,
            )
        return cable, anchor, gripper

    @pytest.fixture
    def world_with_cable_and_gripper(self):
        world = World()
        multi_sim = MujocoSim(world=world, headless=True, step_size=self.STEP_SIZE)
        try:
            multi_sim.start_simulation()
            time.sleep(0.5)
            cable, anchor, gripper = self._create_world_with_full_setup(world)
            yield world, cable, anchor, gripper, multi_sim
        finally:
            multi_sim.stop_simulation()

    def test_cable_reparent_preserves_chain(self, world_with_cable_and_gripper):
        """
        After re-parenting cable_link_0 to the gripper, all cable links
        exist and remain connected.
        """
        import mujoco

        world, cable, anchor, gripper, multi_sim = world_with_cable_and_gripper
        time.sleep(0.5)

        result = multi_sim.simulator.callbacks["attach"](
            body_1_name="cable_link_0",
            body_2_name="gripper",
        )
        assert result.type.value == 1

        time.sleep(0.5)

        body_names = multi_sim.simulator.callbacks["get_all_body_names"]().result
        for i in range(5):
            assert f"cable_link_{i}" in body_names, f"cable_link_{i} missing"

        cable_link_0_id = mujoco.mj_name2id(
            m=multi_sim.simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            name="cable_link_0",
        )
        parent_id = multi_sim.simulator._mj_model.body(cable_link_0_id).parentid[0]
        parent_name = mujoco.mj_id2name(
            m=multi_sim.simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            id=parent_id,
        )
        assert parent_name == "gripper"

        positions = [
            multi_sim.simulator.callbacks["get_body_position"](
                body_name=f"cable_link_{i}"
            ).result
            for i in range(5)
        ]
        for i in range(4):
            dist = np.linalg.norm(positions[i][:3] - positions[i + 1][:3])
            assert dist < 0.2, f"link {i}-{i+1} disconnected: {dist:.3f}"

    def test_cable_reparent_preserves_ball_joints(self, world_with_cable_and_gripper):
        """
        After re-parenting, the ball joints between consecutive links
        still exist in the simulator.
        """
        import mujoco

        world, cable, anchor, gripper, multi_sim = world_with_cable_and_gripper
        time.sleep(0.5)

        multi_sim.simulator.callbacks["attach"](
            body_1_name="cable_link_0",
            body_2_name="gripper",
        )
        time.sleep(0.3)

        for i in range(4):
            joint_name = f"cable_link_{i}_T_cable_link_{i + 1}"
            joint_id = mujoco.mj_name2id(
                m=multi_sim.simulator._mj_model,
                type=mujoco.mjtObj.mjOBJ_JOINT,
                name=joint_name,
            )
            assert joint_id != -1, f"ball joint {joint_name} missing"
            joint_type = multi_sim.simulator._mj_model.joint(joint_id).type[0]
            assert joint_type == mujoco.mjtJoint.mjJNT_BALL

    def test_cable_detach_releases_from_parent(self, world_with_cable_and_gripper):
        """
        Detaching cable_link_0 from the anchor puts it on the worldbody.
        """
        import mujoco

        world, cable, anchor, gripper, multi_sim = world_with_cable_and_gripper
        time.sleep(0.5)

        result = multi_sim.simulator.callbacks["detach"](
            body_name="cable_link_0",
            add_freejoint=True,
        )
        assert result.type.value == 1

        time.sleep(0.3)

        cable_link_0_id = mujoco.mj_name2id(
            m=multi_sim.simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            name="cable_link_0",
        )
        parent_id = multi_sim.simulator._mj_model.body(cable_link_0_id).parentid[0]
        assert parent_id == 0

        body_names = multi_sim.simulator.callbacks["get_all_body_names"]().result
        for i in range(5):
            assert f"cable_link_{i}" in body_names
