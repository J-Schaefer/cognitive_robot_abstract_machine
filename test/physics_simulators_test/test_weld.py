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
