"""
A MuJoCo simulation of a world's rigid bodies with its cables simulated by SuperDex
alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy
from numpy.typing import NDArray
from physics_simulators.mujoco_simulator import MujocoSimulator
from typing_extensions import Any

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.superdex.cable_simulation import (
    BodyPoses,
    SuperdexCableSimulation,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class MujocoBodyPoses(BodyPoses):
    """
    The poses MuJoCo simulates the bodies of a world in; unlike the world's state, they
    show where a servoed joint has got to rather than where it was commanded to.
    """

    simulator: MujocoSimulator
    """
    The simulator the bodies are simulated in.
    """

    def world_T_body(self, body: Body) -> NDArray[numpy.float64]:
        body_data = self.simulator._mj_data.body(body.name.name)
        rotation = numpy.zeros(9)
        mujoco.mju_quat2Mat(rotation, body_data.xquat)
        pose = numpy.eye(4)
        pose[:3, :3] = rotation.reshape(3, 3)
        pose[:3, 3] = body_data.xpos
        return pose


class MujocoSuperdexSim(MujocoSim):
    """
    Simulates the rigid bodies of a world in MuJoCo and its cables in SuperDex; the
    cables touch the collision shapes of a chosen set of bodies.

    Only the stepped simulation advances the cables: every
    :meth:`step_simulation` steps MuJoCo and then the cables over the same period. The
    cables push nothing back onto the rigid bodies.
    """

    cable_simulation: SuperdexCableSimulation
    """
    The simulation of the world's cables.
    """

    def __init__(
        self,
        world: World,
        colliding_bodies: list[Body],
        headless: bool = False,
        step_size: float = 1e-3,
        file_path: Path | None = None,
        **kwargs: Any,
    ):
        """
        :param world: The world to simulate.
        :param colliding_bodies: The bodies whose collision shapes the cables touch.
        :param headless: Whether to run the simulation without a viewer.
        :param step_size: The step size of the MuJoCo simulation, in seconds.
        :param file_path: The file to save the world to, see :class:`MujocoSim`.
        """
        super().__init__(
            world=world,
            headless=headless,
            step_size=step_size,
            file_path=file_path,
            **kwargs,
        )
        # The body poses are only computed once the kinematics has run.
        mujoco.mj_forward(self.simulator._mj_model, self.simulator._mj_data)
        self.cable_simulation = SuperdexCableSimulation(
            world=world,
            colliding_bodies=colliding_bodies,
            body_poses=MujocoBodyPoses(simulator=self.simulator),
        )

    def start_stepped_simulation(self) -> None:
        """
        Start the simulation as :meth:`MujocoSim.start_stepped_simulation` does, with
        the body poses the cables see computed for the state it starts in.
        """
        super().start_stepped_simulation()
        mujoco.mj_forward(self.simulator._mj_model, self.simulator._mj_data)

    def step_simulation(self, duration: timedelta) -> None:
        """
        Advance the rigid bodies as :meth:`MujocoSim.step_simulation` does, then the
        cables over the same period; the viewer draws the cables as they were before the
        step.
        """
        self._draw_cables()
        super().step_simulation(duration)
        self.cable_simulation.step(duration)

    def _draw_cables(self) -> None:
        """
        Draw every cable as a chain of capsules into the viewer's own scene, replacing
        what was drawn there before.
        """
        if self.simulator.headless:
            return
        scene = self.simulator.renderer.user_scn
        scene.ngeom = 0
        for simulated_cable in self.cable_simulation.cables:
            cable = simulated_cable.cable
            rgba = numpy.array(cable.color.to_rgba(), dtype=numpy.float32)
            positions = cable.centreline.corners
            for start, end in pairwise(positions):
                geometry = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geometry,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    numpy.zeros(3),
                    numpy.zeros(3),
                    numpy.eye(3).flatten(),
                    rgba,
                )
                mujoco.mjv_connector(
                    geometry,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    cable.radius,
                    start,
                    end,
                )
                scene.ngeom += 1
