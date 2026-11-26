"""
DAiSy picks a cable up from its table, carries it around and lays it down elsewhere,
with the cable simulated by SuperDex and the robot by MuJoCo.

A straight cable lies on the table, its end reaching a few centimetres past the table's
far edge. The right gripper grips that end from above, lifts the cable off the table,
carries it across the table and lays it down again further along the edge. The gripper
holds the cable by friction alone.

The gripper grips the cable past the table's edge because its fingertips reach 11 mm
beyond the groove between their ridges that holds a cable: gripping a cable lying flat
on the table, the fingertips would hit the table first, and the cable would sit beyond
the ridges, from where its weight pulls it out between the fingers.

Run it as a script; ``--headless`` runs it without the MuJoCo viewer and ``--video``
records it::

    python3 -m experiments.minimal_cable_demo.pick_and_place --headless --video cable.mp4

It needs the ``superdex`` extra of ``semantic_digital_twin`` and the DAiSy description.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
)
from cramph.node import StatechartNode
from experiments.minimal_cable_demo.daisy_cable_scene import (
    CableVideo,
    CommandLineFlag,
    DAiSyCableDemonstration,
)
from numpy.typing import NDArray
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.deformable_cable import DeformableCable
from semantic_digital_twin.world import World

# %% outcome


@dataclass
class PickAndPlaceOutcome:
    """
    Where the cable ended up after it was put down.
    """

    highest_node_height_above_table: float
    """
    How far the highest node of the cable is above the table, in metres; a cable lying
    flat on the table has every node one radius above it.
    """

    distance_from_place_point: float
    """
    How far the point the cable was gripped at lies from where it was to be put down,
    along the table, in metres.
    """


# %% demonstration


@dataclass
class CablePickAndPlaceDemonstration(DAiSyCableDemonstration):
    """
    DAiSy picks a cable up from its table and lays it down elsewhere.
    """

    place_point: tuple[float, float] = (0.635, 0.75)
    """
    Where the gripped point of the cable is laid down, in the frame of the world root,
    in metres; like the pick point, it lies past the table's far edge.
    """

    carry_point: tuple[float, float, float] = (0.3, 0.65, 0.3)
    """
    Where the gripper carries the cable to before it lays it down, in the frame of the
    world root and above the table, in metres.
    """

    clearance: float = 0.15
    """
    How far above the place point the gripper comes before it lays the cable's end down,
    in metres.
    """

    lift_height: float = 0.3
    """
    How far above the table the gripper lifts the cable, in metres.
    """

    settle_duration: timedelta = timedelta(seconds=0.5)
    """
    How long the cable lies on the table before the robot reaches for it, and after it
    was laid down.
    """

    def run(self) -> PickAndPlaceOutcome:
        """
        Set the scene up, pick the cable up, lay it down and measure where it ended up.

        :return: Where the cable ended up.
        """
        world, daisy = self.create_world()
        self.perform_kinematically(daisy, [ParkArmsAction(arms=daisy.all_arms)])
        cable = self.lay_cable_on_the_table(world)
        simulation = self.create_simulation(world)
        try:
            self.let_the_cable_settle(simulation, self.settle_duration)
            self.perform_physically(
                daisy, simulation, self.pick_and_place(daisy, self.pick_point(cable))
            )
            self.let_the_cable_settle(simulation, self.settle_duration)
            return self.outcome(cable, world)
        finally:
            simulation.stop_simulation()
            if self.video is not None:
                self.video.write()

    def pick_and_place(
        self, daisy: DAiSy, pick_point: NDArray[numpy.float64]
    ) -> list[StatechartNode]:
        """
        :param daisy: The robot.
        :param pick_point: Where the gripper grips the cable.
        :return: The steps picking the cable up at the pick point, carrying it across
            the table and laying it down with the pick point at :attr:`place_point`.
        """
        arm = daisy.right_arm
        table_top = self.table_top(daisy._world)
        place_point = numpy.array([*self.place_point, pick_point[2]])
        grip_points = [
            numpy.array([*pick_point[:2], table_top + self.lift_height]),
            numpy.array(self.carry_point) + numpy.array([0.0, 0.0, table_top]),
            place_point + numpy.array([0.0, 0.0, self.clearance]),
            place_point,
        ]
        return [
            *self.pick_up(arm, pick_point),
            *[self.move_from_above(arm, point) for point in grip_points],
            self.open_gripper(arm),
            self.move_from_above(
                arm, place_point - self.approach_distance * self.pointing
            ),
        ]

    def outcome(self, cable: DeformableCable, world: World) -> PickAndPlaceOutcome:
        """
        :param cable: The cable.
        :param world: The world.
        :return: Where the cable ended up.
        """
        nodes = cable.centreline.corners
        picked = self.pick_point(cable)
        return PickAndPlaceOutcome(
            highest_node_height_above_table=float(
                nodes[:, 2].max() - self.table_top(world)
            ),
            distance_from_place_point=float(
                numpy.linalg.norm(picked[:2] - numpy.array(self.place_point))
            ),
        )


def run_from_command_line() -> None:
    """
    Parse the command line and run the demonstration.
    """
    parser = argparse.ArgumentParser(
        description="DAiSy picks a cable up from its table and puts it down elsewhere."
    )
    parser.add_argument(
        CommandLineFlag.HEADLESS,
        action="store_true",
        help="Run without the MuJoCo viewer.",
    )
    parser.add_argument(
        CommandLineFlag.VIDEO, type=Path, help="Record a video into this file."
    )
    arguments = parser.parse_args()
    outcome = CablePickAndPlaceDemonstration(
        headless=arguments.headless,
        video=None if arguments.video is None else CableVideo(path=arguments.video),
    ).run()
    print(outcome)


if __name__ == "__main__":
    run_from_command_line()
