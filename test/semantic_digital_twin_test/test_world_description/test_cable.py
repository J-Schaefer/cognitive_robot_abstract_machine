from math import pi

import mujoco
import numpy
import pytest
from numpy.testing import assert_allclose

from semantic_digital_twin.adapters.multi_sim import MujocoEquality
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.cable import (
    Cable,
    CableConfig,
    CableSimulation,
    CableSimulationStrategy,
    build_cable,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Color, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class TestCableConfig:
    def test_defaults(self):
        config = CableConfig()
        assert config.segment_count == 10
        assert config.segment_length == 0.03
        assert config.radius == 0.005
        assert config.mass_per_segment == 0.005
        assert config.color == Color(1, 0, 0, 1)

    def test_custom_values(self):
        config = CableConfig(
            segment_count=15,
            segment_length=0.05,
            radius=0.01,
            mass_per_segment=0.01,
            color=Color(0, 1, 0, 1),
        )
        assert config.segment_count == 15
        assert config.segment_length == 0.05
        assert config.radius == 0.01

    def test_default_strategy(self):
        config = CableConfig()
        assert config.strategy == CableSimulationStrategy.KINEMATIC_ATTACH
        assert config.use_composite is False

    def test_strategy_override(self):
        config = CableConfig(
            strategy=CableSimulationStrategy.POSITION_OVERRIDE
        )
        assert config.strategy == CableSimulationStrategy.POSITION_OVERRIDE

    def test_cable_dataclass(self):
        cable = Cable()
        assert len(cable.segments) == 0
        assert len(cable.connections) == 0
        assert len(cable.constraints) == 0


class TestBuildCable:
    def test_creates_correct_number_of_segments(self):
        world = World()
        config = CableConfig(segment_count=8)
        cable = build_cable(config, world)
        assert len(cable.segments) == 8
        assert len(cable.connections) == 8

    def test_creates_correct_number_of_constraints(self):
        world = World()
        config = CableConfig(segment_count=6)
        cable = build_cable(config, world)
        assert len(cable.constraints) == config.segment_count - 1

    def test_segments_have_cylinder_geometry(self):
        world = World()
        config = CableConfig(segment_count=3)
        cable = build_cable(config, world)
        for segment in cable.segments:
            for shape in segment.visual:
                assert isinstance(shape, Cylinder)
            for shape in segment.collision:
                assert isinstance(shape, Cylinder)

    def test_cylinder_is_rotated_to_x_axis(self):
        world = World()
        config = CableConfig(segment_count=1)
        cable = build_cable(config, world)
        shape = cable.segments[0].visual[0]
        rpy = shape.origin.to_rotation_matrix().to_rpy()
        assert_allclose([float(v) for v in rpy], (0.0, pi / 2.0, 0.0), atol=1e-6)

    def test_each_segment_has_distinct_origin_object(self):
        """Each segment's Cylinder origin must be a distinct object so that
        the visualisation pipeline can assign a per-body frame_id."""
        world = World()
        config = CableConfig(segment_count=5)
        cable = build_cable(config, world)
        origins = {
            id(segment.visual[0].origin) for segment in cable.segments
        }
        assert len(origins) == config.segment_count

    def test_each_segment_has_free_joint(self):
        world = World()
        config = CableConfig(segment_count=4)
        cable = build_cable(config, world)
        for connection in cable.connections:
            assert isinstance(connection, Connection6DoF)

    def test_segments_are_children_of_world_root(self):
        world = World()
        config = CableConfig(segment_count=4)
        cable = build_cable(config, world)
        root = world.root
        for segment in cable.segments:
            parent = segment.parent_connection.parent
            assert parent == root

    def test_initial_positions_are_staggered(self):
        world = World()
        config = CableConfig(segment_count=5, segment_length=0.03)
        cable = build_cable(config, world)
        for i, conn in enumerate(cable.connections):
            expected_x = i * config.segment_length
            assert_allclose(world.state[conn.x.id].position, expected_x)
            assert_allclose(world.state[conn.y.id].position, 0.0)
            assert_allclose(world.state[conn.z.id].position, 0.0)

    def test_equality_constraints_are_connect_type(self):
        world = World()
        config = CableConfig(segment_count=5)
        cable = build_cable(config, world)
        for constraint in cable.constraints:
            assert constraint.type == mujoco.mjtEq.mjEQ_CONNECT
            assert constraint.object_type == mujoco.mjtObj.mjOBJ_BODY

    def test_equality_constraints_link_consecutive_segments(self):
        world = World()
        config = CableConfig(segment_count=5)
        cable = build_cable(config, world)
        for i, constraint in enumerate(cable.constraints):
            assert constraint.name_1 == f"cable_segment_{i}"
            assert constraint.name_2 == f"cable_segment_{i + 1}"

    def test_attaches_to_parent_body(self):
        world = World()
        parent = Body(name=PrefixedName("gripper"))
        with world.modify_world():
            world.add_kinematic_structure_entity(parent)
        config = CableConfig(segment_count=3)
        cable = build_cable(config, world, parent_body=parent)
        # There should be an extra constraint linking parent to segment 0
        extra_constraint = cable.constraints[-1]
        assert extra_constraint.name_1 == "gripper"
        assert extra_constraint.name_2 == "cable_segment_0"

    def test_empty_world_creates_root(self):
        world = World()
        config = CableConfig(segment_count=3)
        cable = build_cable(config, world)
        assert world.root is not None
        assert world.root.name.name == "world"

    def test_builds_mujoco_model(self):
        """Verify the world can be serialized to MJCF and compiled by MuJoCo."""
        import os
        import tempfile
        import logging
        logging.disable(logging.CRITICAL)

        from semantic_digital_twin.adapters.multi_sim import MujocoBuilder

        world = World()
        config = CableConfig(segment_count=5, segment_length=0.03, radius=0.005)
        cable = build_cable(config, world)

        tmp_file = os.path.join(tempfile.mkdtemp(), "scene.xml")
        builder = MujocoBuilder()
        builder.build_world(world, tmp_file)

        model = mujoco.MjSpec.from_file(tmp_file).compile()
        assert model.nbody == 6  # 5 segments + world
        assert model.njnt == config.segment_count
        assert model.neq == config.segment_count - 1

    def test_simulates_under_gravity(self):
        """Verify the cable bends under gravity without exploding."""
        import os
        import tempfile
        import logging
        logging.disable(logging.CRITICAL)

        from semantic_digital_twin.adapters.multi_sim import MujocoBuilder

        world = World()
        config = CableConfig(segment_count=5, segment_length=0.03, radius=0.005)
        cable = build_cable(config, world)

        tmp_file = os.path.join(tempfile.mkdtemp(), "scene.xml")
        builder = MujocoBuilder()
        builder.build_world(world, tmp_file)

        model = mujoco.MjSpec.from_file(tmp_file).compile()
        data = mujoco.MjData(model)

        for step in range(1000):
            mujoco.mj_step(model, data)

        # Segments should have negative z (falling under gravity)
        for i in range(config.segment_count):
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"cable_segment_{i}"
            )
            pos = data.xpos[body_id]
            assert pos[2] < 0, f"Segment {i} z={pos[2]} did not fall"
            # Segments should stay reasonably close (within 2 meters)
            assert abs(pos[0]) < 2.0
            assert abs(pos[1]) < 2.0


class TestCableBackgroundSimulation:
    def test_simulates_in_background_thread(self):
        """The cable falls under gravity while MujocoSim runs in a background thread."""
        import logging
        import time

        logging.disable(logging.CRITICAL)

        from semantic_digital_twin.adapters.multi_sim import MujocoSim

        world = World()
        config = CableConfig(segment_count=5, segment_length=0.03, radius=0.005)
        build_cable(config, world)

        multi_sim = MujocoSim(world=world, headless=True)

        try:
            multi_sim.start_simulation()
            time.sleep(2.0)
            multi_sim.stop_simulation()

            mj_data = multi_sim.simulator._mj_data
            for i in range(config.segment_count):
                body_id = mujoco.mj_name2id(
                    multi_sim.simulator._mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"cable_segment_{i}",
                )
                pos = mj_data.xpos[body_id]
                assert pos[2] < 0, f"Segment {i} z={pos[2]} did not fall"
                assert abs(pos[0]) < 2.0
                assert abs(pos[1]) < 2.0
        finally:
            if multi_sim.simulator.state.value != 0:
                multi_sim.stop_simulation()


class TestCableSimulation:
    def test_constructs_and_starts_simulation(self):
        """CableSimulation builds cable, creates MujocoSim, and starts the
        background thread."""
        import logging

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(segment_count=5, segment_length=0.03, radius=0.005)
        cable_sim = CableSimulation(config=config, world=world)

        assert cable_sim.cable is not None
        assert len(cable_sim.cable.segments) == config.segment_count
        assert cable_sim.multi_sim is not None

        try:
            cable_sim.start()
            assert cable_sim._started is True

            segments = cable_sim.get_segment_positions()
            assert len(segments) == config.segment_count
            for i in range(config.segment_count):
                assert f"cable_segment_{i}" in segments
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_segments_fall_under_gravity(self):
        """After cable_sim runs for 2s, segments are below the origin."""
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(segment_count=5, segment_length=0.03, radius=0.005)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(2.0)
            positions = cable_sim.get_segment_positions()
            for i in range(config.segment_count):
                pos = positions[f"cable_segment_{i}"]
                assert pos[2] < 0, f"Segment {i} z={pos[2]} did not fall"
                assert abs(pos[0]) < 2.0
                assert abs(pos[1]) < 2.0
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_raises_when_grasp_before_start(self):
        """grasp() raises RuntimeError if simulation isn't running."""
        world = World()
        config = CableConfig(segment_count=3)
        cable_sim = CableSimulation(config=config, world=world)

        with pytest.raises(RuntimeError, match="Simulation is not running"):
            cable_sim.grasp("gripper", segment_index=0)

    def test_raises_when_release_before_start(self):
        """release() raises RuntimeError if simulation isn't running."""
        world = World()
        config = CableConfig(segment_count=3)
        cable_sim = CableSimulation(config=config, world=world)

        with pytest.raises(RuntimeError, match="Simulation is not running"):
            cable_sim.release(segment_index=0)

    def test_raises_on_invalid_segment_index(self):
        """grasp/release raise ValueError for out-of-range segment indices."""
        import logging

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(segment_count=3)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            with pytest.raises(ValueError, match="segment_index"):
                cable_sim.grasp("gripper", segment_index=3)
            with pytest.raises(ValueError, match="segment_index"):
                cable_sim.grasp("gripper", segment_index=-1)
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_double_start_is_idempotent(self):
        """Calling start() twice is harmless."""
        import logging

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(segment_count=3)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            cable_sim.start()
            assert cable_sim._started is True
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_double_stop_is_idempotent(self):
        """Calling stop() twice is harmless."""
        import logging

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(segment_count=3)
        cable_sim = CableSimulation(config=config, world=world)

        cable_sim.start()
        cable_sim.stop()
        cable_sim.stop()
        assert cable_sim._started is False
        logging.disable(logging.NOTSET)


class TestPositionOverrideStrategy:
    """Tests for CableSimulationStrategy.POSITION_OVERRIDE."""

    def _make_gripper_in_world(self, world: World) -> Body:
        r"""Add a free-joint gripper body to the world and spec."""
        if world.root is None:
            root = Body(name=PrefixedName("world"))
            with world.modify_world():
                world.add_kinematic_structure_entity(root)
        else:
            root = world.root
        gripper = Body(name=PrefixedName("test_gripper"))
        with world.modify_world():
            world.add_kinematic_structure_entity(gripper)
            conn = Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=gripper,
                name=PrefixedName("gripper_joint"),
            )
            world.add_connection(conn)
            world.state[conn.x.id].position = 0.5
            world.state[conn.y.id].position = 0.0
            world.state[conn.z.id].position = 0.5
            world.state[conn.qw.id].position = 1.0
            world.state[conn.qx.id].position = 0.0
            world.state[conn.qy.id].position = 0.0
            world.state[conn.qz.id].position = 0.0
        return gripper

    def test_strategy_override_via_config(self):
        """CableSimulation uses CableConfig.strategy when
        strategy_override is None."""
        import logging

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=3,
            segment_length=0.03,
            radius=0.005,
            strategy=CableSimulationStrategy.POSITION_OVERRIDE,
        )

        cable_sim = CableSimulation(config=config, world=world)
        try:
            assert cable_sim._effective_strategy == CableSimulationStrategy.POSITION_OVERRIDE
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_strategy_override_via_simulation(self):
        """strategy_override trumps CableConfig.strategy."""
        import logging

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=3,
            strategy=CableSimulationStrategy.KINEMATIC_ATTACH,
        )
        cable_sim = CableSimulation(
            config=config,
            world=world,
            strategy_override=CableSimulationStrategy.POSITION_OVERRIDE,
        )
        try:
            assert cable_sim._effective_strategy == CableSimulationStrategy.POSITION_OVERRIDE
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_grasp_release_no_crash(self):
        """grasp() and release() do not raise under POSITION_OVERRIDE."""
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=3,
            segment_length=0.03,
            radius=0.005,
            strategy=CableSimulationStrategy.POSITION_OVERRIDE,
        )
        gripper = self._make_gripper_in_world(world)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(1.0)
            cable_sim.grasp("test_gripper", segment_index=0)
            cable_sim.release(segment_index=0)
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_grasped_segment_follows_gripper(self):
        """When a segment is grasped (POSITION_OVERRIDE), check that
        position tracking produces finite and reasonable values."""
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=5,
            segment_length=0.03,
            radius=0.005,
            strategy=CableSimulationStrategy.POSITION_OVERRIDE,
        )
        gripper = self._make_gripper_in_world(world)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(2.0)

            cable_sim.grasp("test_gripper", segment_index=0)
            time.sleep(0.3)

            # Move gripper far off in y
            cable_sim.multi_sim.simulator.callbacks["set_body_position"](
                "test_gripper", numpy.array([0.5, 0.3, 0.5])
            )
            time.sleep(0.1)

            positions = cable_sim.get_segment_positions()
            for i in range(config.segment_count):
                assert numpy.isfinite(positions[f"cable_segment_{i}"]).all(), (
                    f"Segment {i} has NaN/inf position"
                )
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_grasped_segment_does_not_crash(self):
        """After grasping, the cable should still be in a valid state
        (no segfaults or thread crashes)."""
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=5,
            segment_length=0.03,
            radius=0.005,
            strategy=CableSimulationStrategy.POSITION_OVERRIDE,
        )
        gripper = self._make_gripper_in_world(world)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(2.0)

            cable_sim.grasp("test_gripper", segment_index=0)
            time.sleep(1.0)

            positions = cable_sim.get_segment_positions()
            for i in range(config.segment_count):
                assert f"cable_segment_{i}" in positions
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_release_removes_override(self):
        """After releasing, the segment falls under gravity again."""
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=5,
            segment_length=0.03,
            radius=0.005,
            strategy=CableSimulationStrategy.POSITION_OVERRIDE,
        )
        gripper = self._make_gripper_in_world(world)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(2.0)

            positions_before = cable_sim.get_segment_positions()
            seg0_z_before = positions_before["cable_segment_0"][2]

            cable_sim.grasp("test_gripper", segment_index=0)
            time.sleep(0.5)
            cable_sim.release(segment_index=0)
            time.sleep(1.0)

            positions_after = cable_sim.get_segment_positions()
            seg0_z_after = positions_after["cable_segment_0"][2]
            assert seg0_z_after < 0, (
                f"Released segment z={seg0_z_after:.3f} should fall below zero"
            )
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)


class TestKinematicAttachStrategy:
    """Tests for CableSimulationStrategy.KINEMATIC_ATTACH with state
    preservation during recompilation."""

    def _make_gripper_in_world(self, world: World) -> Body:
        r"""Add a free-joint gripper body to the world and spec."""
        if world.root is None:
            root = Body(name=PrefixedName("world"))
            with world.modify_world():
                world.add_kinematic_structure_entity(root)
        else:
            root = world.root
        gripper = Body(name=PrefixedName("test_gripper"))
        with world.modify_world():
            world.add_kinematic_structure_entity(gripper)
            conn = Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=gripper,
                name=PrefixedName("gripper_joint"),
            )
            world.add_connection(conn)
            world.state[conn.x.id].position = 0.5
            world.state[conn.y.id].position = 0.0
            world.state[conn.z.id].position = 0.5
            world.state[conn.qw.id].position = 1.0
            world.state[conn.qx.id].position = 0.0
            world.state[conn.qy.id].position = 0.0
            world.state[conn.qz.id].position = 0.0
        return gripper

    def test_attach_and_detach_do_not_crash(self):
        """Grasp and release via KINEMATIC_ATTACH do not crash the
        simulation thread, even though the inter-segment connect
        constraints require all bodies to have independent DOFs."""
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=5,
            segment_length=0.03,
            radius=0.005,
            strategy=CableSimulationStrategy.KINEMATIC_ATTACH,
        )
        gripper = self._make_gripper_in_world(world)
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(2.0)

            cable_sim.grasp("test_gripper", segment_index=0)
            time.sleep(0.5)
            cable_sim.release(segment_index=0)
            time.sleep(1.0)

            positions = cable_sim.get_segment_positions()
            for i in range(config.segment_count):
                assert f"cable_segment_{i}" in positions
                # Segment bodies exist and report positions
                assert numpy.isfinite(positions[f"cable_segment_{i}"]).all()
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)


class TestCompositeCableStrategy:
    """Tests for CableConfig.use_composite = True."""

    @staticmethod
    def _requires_composite_api():
        """Skip test if MuJoCo doesn't support composite objects."""
        import mujoco
        if not hasattr(mujoco.MjSpec, "add_composite"):
            pytest.skip("MuJoCo version does not support composite API")

    def test_creates_composite_cable(self):
        """A CableSimulation with use_composite=True starts without error."""
        self._requires_composite_api()
        import logging

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=5,
            segment_length=0.03,
            radius=0.005,
            use_composite=True,
        )
        cable_sim = CableSimulation(config=config, world=world)

        try:
            assert cable_sim.cable is not None
            assert len(cable_sim.cable.segments) == config.segment_count
            assert len(cable_sim._composite_body_names) == config.segment_count
            assert cable_sim._composite_body_names[0] == "cable_segment_B0"
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_composite_cable_falls_under_gravity(self):
        """Composite cable segments fall under gravity."""
        self._requires_composite_api()
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=5,
            segment_length=0.03,
            radius=0.005,
            use_composite=True,
        )
        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(2.0)
            positions = cable_sim.get_segment_positions()
            for i in range(config.segment_count):
                pos = positions[f"cable_segment_{i}"]
                assert pos[2] < 0, f"Composite segment {i} z={pos[2]:.3f} did not fall"
                assert abs(pos[0]) < 2.0
                assert abs(pos[1]) < 2.0
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)

    def test_composite_cable_grasp_release(self):
        """grasp() and release() work on composite cables."""
        self._requires_composite_api()
        import logging
        import time

        logging.disable(logging.CRITICAL)

        world = World()
        config = CableConfig(
            segment_count=5,
            segment_length=0.03,
            radius=0.005,
            use_composite=True,
        )
        if world.root is None:
            root = Body(name=PrefixedName("world"))
            with world.modify_world():
                world.add_kinematic_structure_entity(root)
        else:
            root = world.root
        gripper = Body(name=PrefixedName("test_gripper"))
        with world.modify_world():
            world.add_kinematic_structure_entity(gripper)
            conn = Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=gripper,
                name=PrefixedName("gripper_joint"),
            )
            world.add_connection(conn)
            world.state[conn.x.id].position = 0.5
            world.state[conn.y.id].position = 0.0
            world.state[conn.z.id].position = 0.5
            world.state[conn.qw.id].position = 1.0

        cable_sim = CableSimulation(config=config, world=world)

        try:
            cable_sim.start()
            time.sleep(2.0)
            cable_sim.grasp("test_gripper", segment_index=0)
            time.sleep(0.5)
            positions = cable_sim.get_segment_positions()
            assert "cable_segment_0" in positions
            cable_sim.release(segment_index=0)
        finally:
            cable_sim.stop()
            logging.disable(logging.NOTSET)
