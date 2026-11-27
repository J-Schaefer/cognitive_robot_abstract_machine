"""
Following a running Giskard in the physics of the cable demonstrations, with Giskard
stood in for by an executor that moves a joint of its world.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta

import pytest
from semantic_digital_twin.robots.daisy import DAiSy

pytest.importorskip("superdex.physics")

from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.minimal_cable_demo.daisy_cable_scene import CommandLineFlag
from experiments.minimal_cable_demo.demo import MinimalCableDemonstration
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
)

pytestmark = pytest.mark.skipif(
    not DAiSy.is_description_installed(), reason="DAiSy not installed"
)

# %% a stand-in for the running Giskard


@dataclass(kw_only=True)
class JointMovingGiskard:
    """
    Stands in for the executor sending a plan to a running Giskard: executing it
    commands one joint of the world Giskard serves to a new position.
    """

    context: StatechartContext
    """
    The context of the plan, over the world Giskard serves.
    """

    joint: DegreeOfFreedom
    """
    The degree of freedom of the joint that is moved.
    """

    target_position: float
    """
    Where the joint is moved to.
    """

    def compile(self, *, statechart: Statechart) -> None:
        """
        Accept the plan without looking at it.
        """

    def execute(self) -> None:
        """
        Command the joint to :attr:`target_position`.
        """
        world = self.context.world
        world.state[self.joint.id].position = self.target_position
        world.notify_state_change()

    def close(self) -> None:
        """
        Nothing to release.
        """


# %% following a running Giskard


def test_the_simulated_robot_is_commanded_where_the_running_giskard_moved_the_robot():
    """
    Once a step sent to the running Giskard is done, the simulated robot is commanded to
    the joint positions the world Giskard serves holds.
    """
    demonstration = MinimalCableDemonstration(
        headless=True, uses_running_giskard=True, settle_duration=timedelta()
    )
    world, _ = MinimalCableDemonstration(headless=True).create_world()
    demonstration.giskard_world, giskard_daisy = MinimalCableDemonstration(
        headless=True
    ).create_world()
    [finger] = giskard_daisy.right_arm.end_effector.finger_connections
    target_position = 0.02
    demonstration._running_giskard_executor = lambda *, daisy: JointMovingGiskard(
        context=StatechartContext(world=demonstration.giskard_world),
        joint=finger.raw_dof,
        target_position=target_position,
    )
    simulation = demonstration.create_simulation(world=world)
    try:
        demonstration.perform_physically(
            daisy=giskard_daisy,
            simulation=simulation,
            steps=[
                demonstration.close_on_cable(
                    gripper=giskard_daisy.left_arm.end_effector
                )
            ],
        )
    finally:
        simulation.stop_simulation()

    assert world.state[
        world.get_degree_of_freedom_by_name(finger.raw_dof.name).id
    ].position == pytest.approx(target_position)


# %% command line


def test_the_running_giskard_switch_makes_the_demonstration_use_the_running_giskard(
    monkeypatch,
):
    """
    The ``--running-giskard`` switch makes the demonstration send the robot's motions to
    a Giskard node already running.
    """
    monkeypatch.setattr(sys, "argv", ["demo", CommandLineFlag.RUNNING_GISKARD])

    demonstration = MinimalCableDemonstration.from_command_line(description="")

    assert demonstration.uses_running_giskard
