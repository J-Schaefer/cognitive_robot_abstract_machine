from __future__ import annotations

import numpy as np
import pytest

from physics_simulators.mujoco_simulator import MujocoSimulator
from physics_simulators.base_simulator import (
    SimulatorCallbackResult,
)


class TestWeldBodies:
    """
    Tests for weld_bodies and unweld_bodies callbacks.
    """

    @pytest.fixture
    def simulator(self):
        import mujoco
        import tempfile
        import os

        xml = """<mujoco model="weld_test">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="body1" pos="0 0 0.5" quat="1 0 0 0">
      <freejoint/>
      <geom name="body1_geom" type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>
    <body name="body2" pos="0.1 0 0.5" quat="1 0 0 0">
      <freejoint/>
      <geom name="body2_geom" type="sphere" size="0.03" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            file_path = f.name

        sim = MujocoSimulator(
            _headless=True,
            _step_size=0.001,
            file_path=file_path,
        )
        yield sim
        try:
            sim.stop()
        except Exception:
            pass
        try:
            os.unlink(file_path)
        except Exception:
            pass

    def test_weld_makes_bodies_move_together(self, simulator):
        """
        Welded bodies maintain constant relative distance.
        """
        simulator.start(simulate_in_thread=False, render_in_thread=False)

        for _ in range(50):
            simulator.step()

        pos1_before = simulator.callbacks["get_body_position"](body_name="body1").result
        pos2_before = simulator.callbacks["get_body_position"](body_name="body2").result
        rel_before = np.linalg.norm(pos1_before[:3] - pos2_before[:3])

        result = simulator.callbacks["weld_bodies"](
            body_1_name="body1",
            body_2_name="body2",
        )
        assert (
            result.type
            == SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )

        for _ in range(50):
            simulator.step()

        pos1_after = simulator.callbacks["get_body_position"](body_name="body1").result
        pos2_after = simulator.callbacks["get_body_position"](body_name="body2").result
        rel_after = np.linalg.norm(pos1_after[:3] - pos2_after[:3])

        assert abs(rel_after - rel_before) < 0.001

        simulator.stop()

    def test_weld_idempotent(self, simulator):
        """
        Welding already-welded bodies returns SUCCESS_WITHOUT_EXECUTION.
        """
        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()

        simulator.callbacks["weld_bodies"](
            body_1_name="body1",
            body_2_name="body2",
        )
        result = simulator.callbacks["weld_bodies"](
            body_1_name="body1",
            body_2_name="body2",
        )
        assert (
            result.type == SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        )
        simulator.stop()

    def test_unweld_restores_independent_motion(self, simulator):
        """
        After unwelding, bodies move independently again.
        """
        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()

        simulator.callbacks["weld_bodies"](
            body_1_name="body1",
            body_2_name="body2",
        )
        result = simulator.callbacks["unweld_bodies"](
            body_1_name="body1",
            body_2_name="body2",
        )
        assert (
            result.type
            == SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )

        for _ in range(50):
            simulator.step()

        pos1 = simulator.callbacks["get_body_position"](body_name="body1").result
        pos2 = simulator.callbacks["get_body_position"](body_name="body2").result
        rel_dist = np.linalg.norm(pos1[:3] - pos2[:3])
        assert rel_dist > 0.005
        simulator.stop()

    def test_cannot_weld_same_body(self, simulator):
        """
        Welding a body to itself is rejected.
        """
        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()
        result = simulator.callbacks["weld_bodies"](
            body_1_name="body1",
            body_2_name="body1",
        )
        assert (
            result.type
            == SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
        )
        simulator.stop()

    def test_cannot_weld_nonexistent_body(self, simulator):
        """
        Welding a nonexistent body returns failure.
        """
        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()
        result = simulator.callbacks["weld_bodies"](
            body_1_name="does_not_exist",
            body_2_name="body1",
        )
        assert (
            result.type
            == SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
        )
        simulator.stop()


class TestRecursiveAttachDetach:
    """
    Tests for attach and detach callbacks with bodies that have child bodies.
    """

    @pytest.fixture
    def simulator(self):
        import mujoco
        import tempfile
        import os

        xml = """<mujoco model="attach_test">
  <worldbody>
    <body name="parent" pos="0 0 1" quat="1 0 0 0">
      <geom name="parent_geom" type="sphere" size="0.05" rgba="1 0 0 1"/>
      <body name="child" pos="0 0.1 0" quat="1 0 0 0">
        <geom name="child_geom" type="sphere" size="0.05" rgba="1 0 0 1"/>
        <body name="grandchild" pos="0 0 0.1" quat="1 0 0 0">
          <geom name="grandchild_geom" type="sphere" size="0.03" rgba="0 1 0 1"/>
          <joint name="grandchild_joint" type="ball" pos="0 0 -0.1"/>
        </body>
      </body>
    </body>
    <body name="target" pos="0 2 1" quat="1 0 0 0">
      <geom name="target_geom" type="sphere" size="0.05" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            file_path = f.name

        sim = MujocoSimulator(
            _headless=True,
            _step_size=0.001,
            file_path=file_path,
        )
        yield sim
        try:
            sim.stop()
        except Exception:
            pass
        try:
            os.unlink(file_path)
        except Exception:
            pass

    def test_attach_preserves_child_bodies(self, simulator):
        """
        Attaching a body with children preserves the entire subtree.
        """
        import mujoco

        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()

        result = simulator.callbacks["attach"](
            body_1_name="child",
            body_2_name="target",
        )
        assert (
            result.type
            == SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )

        body_names = simulator.callbacks["get_all_body_names"]().result
        assert "child" in body_names
        assert "grandchild" in body_names

        grandchild_id = mujoco.mj_name2id(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            name="grandchild",
        )
        assert grandchild_id != -1

        grandchild_pos = simulator.callbacks["get_body_position"](
            body_name="grandchild"
        ).result
        assert grandchild_pos is not None

        simulator.stop()

    def test_attach_child_body_has_new_parent(self, simulator):
        """
        After attach, the moved child body's parent is the target.
        """
        import mujoco

        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()

        simulator.callbacks["attach"](
            body_1_name="child",
            body_2_name="target",
        )

        child_id = mujoco.mj_name2id(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            name="child",
        )
        parent_id = simulator._mj_model.body(child_id).parentid[0]
        parent_name = mujoco.mj_id2name(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            id=parent_id,
        )
        assert parent_name == "target"
        simulator.stop()

    def test_attach_grandchild_remains_child_of_child(self, simulator):
        """
        After attaching child to target, grandchild remains a child of child.
        """
        import mujoco

        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()

        simulator.callbacks["attach"](
            body_1_name="child",
            body_2_name="target",
        )

        grandchild_id = mujoco.mj_name2id(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            name="grandchild",
        )
        parent_id = simulator._mj_model.body(grandchild_id).parentid[0]
        parent_name = mujoco.mj_id2name(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            id=parent_id,
        )
        assert parent_name == "child"
        simulator.stop()

    def test_attach_preserves_grandchild_ball_joint(self, simulator):
        """
        After attaching child, the grandchild's ball joint still exists.
        """
        import mujoco

        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()

        simulator.callbacks["attach"](
            body_1_name="child",
            body_2_name="target",
        )

        grandchild_joint_id = mujoco.mj_name2id(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_JOINT,
            name="grandchild_joint",
        )
        assert grandchild_joint_id != -1
        joint_type = simulator._mj_model.joint(grandchild_joint_id).type[0]
        assert joint_type == mujoco.mjtJoint.mjJNT_BALL
        simulator.stop()

    def test_detach_preserves_child_bodies(self, simulator):
        """
        Detaching a body with children preserves the entire subtree.
        """
        import mujoco

        simulator.start(simulate_in_thread=False, render_in_thread=False)
        for _ in range(10):
            simulator.step()

        result = simulator.callbacks["detach"](
            body_name="child",
            add_freejoint=True,
        )
        assert (
            result.type
            == SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )

        body_names = simulator.callbacks["get_all_body_names"]().result
        assert "child" in body_names
        assert "grandchild" in body_names

        grandchild_id = mujoco.mj_name2id(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            name="grandchild",
        )
        assert grandchild_id != -1

        child_id = mujoco.mj_name2id(
            m=simulator._mj_model,
            type=mujoco.mjtObj.mjOBJ_BODY,
            name="child",
        )
        parent_id = simulator._mj_model.body(child_id).parentid[0]
        assert parent_id == 0
        simulator.stop()
