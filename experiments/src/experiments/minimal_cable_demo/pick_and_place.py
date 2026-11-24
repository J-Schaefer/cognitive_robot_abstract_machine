"""
DAiSy picks a cable up from its table, carries it around and lays it down elsewhere, with
the cable simulated by SuperDex and the robot by MuJoCo.

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
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
    SetGripperAction,
)
from cramph.node import StatechartNode
from experiments.minimal_cable_demo.daisy_cable_scene import (
    CableVideo,
    CommandLineFlag,
    DAiSyCableDemonstration,
)
from krrood_spatial_math.spatial_types import Point3
from numpy.typing import NDArray
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import Arm
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

    cable_end: tuple[float, float] = (0.66, 0.45)
    """
    Where the end of the cable lies, 6 cm past the table's far edge, in the frame of
    the world root, in metres; the cable lies from there along the world's negative x
    axis.
    """

    pick_arc_length: float = 0.025
    """
    How far from its end the gripper grips the cable, in metres: far enough past the
    table's edge that the fingertips pass it.
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

    approach_distance: float = 0.1
    """
    How far above the cable the gripper stops before it reaches down for it, and how
    far it backs off once it let go, in metres.
    """

    clearance: float = 0.15
    """
    How far above the place point the gripper comes before it lays the cable's end
    down, in metres.
    """

    lift_height: float = 0.3
    """
    How far above the table the gripper lifts the cable, in metres.
    """

    approach_position_threshold: float = 1.0e-3
    """
    How close, in metres, the tool frame has to come to where it grips the cable.
    """

    settle_duration: timedelta = timedelta(seconds=0.5)
    """
    How long the cable lies on the table before the robot reaches for it, and after it
    was laid down.
    """

    pointing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([0.0, 0.0, -1.0])
    )
    """
    The direction the gripper points along: straight down.
    """

    closing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([0.0, 1.0, 0.0])
    )
    """
    The direction the gripper's fingers close along: across the cable.
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

    def lay_cable_on_the_table(self, world: World) -> DeformableCable:
        """
        Add a straight cable to the world, lying on the table along the world's x axis
        with its end past the table's far edge.

        :param world: The world to add the cable to.
        :return: The cable.
        """
        end = numpy.array([*self.cable_end, self.table_top(world) + self.cable_radius])
        start = end - numpy.array([self.cable_length, 0.0, 0.0])
        cable = DeformableCable.along_polyline(
            name=PrefixedName("cable"),
            polyline=[
                Point3(*start, reference_frame=world.root),
                Point3(*end, reference_frame=world.root),
            ],
            number_of_segments=self.number_of_cable_segments,
            radius=self.cable_radius,
            color=self.cable_color,
        )
        with world.modify_world():
            world.add_semantic_annotation(cable)
        return cable

    def pick_point(self, cable: DeformableCable) -> NDArray[numpy.float64]:
        """
        :param cable: The cable.
        :return: Where the gripper grips the cable, as the cable lies now.
        """
        return cable.point_at_arc_length(
            cable.centreline.length - self.pick_arc_length
        ).to_np()[:3]

    def pick_and_place(
        self, daisy: DAiSy, pick_point: NDArray[numpy.float64]
    ) -> list[StatechartNode]:
        """
        :param daisy: The robot.
        :param pick_point: Where the gripper grips the cable.
        :return: The steps picking the cable up at the pick point, carrying it across
            the table and laying it down with the pick point at :attr:`place_point`.
        """
        world = daisy._world
        arm = daisy.right_arm
        table_top = self.table_top(world)
        place_point = numpy.array([*self.place_point, pick_point[2]])
        before_pick = pick_point - self.approach_distance * self.pointing
        open_gripper = arm.end_effector.default_configuration(GripperState.OPEN)
        moves_to_pick = [self._move_to(arm, world, before_pick)]
        grip_points = [
            numpy.array([*pick_point[:2], table_top + self.lift_height]),
            numpy.array(self.carry_point) + numpy.array([0.0, 0.0, table_top]),
            place_point + numpy.array([0.0, 0.0, self.clearance]),
            place_point,
        ]
        moves_to_place = [self._move_to(arm, world, point) for point in grip_points]
        return [
            SetGripperAction(gripper_configuration=open_gripper),
            *moves_to_pick,
            self._move_to(arm, world, pick_point, self.approach_position_threshold),
            self.close_on_cable(arm.end_effector),
            *moves_to_place,
            SetGripperAction(gripper_configuration=open_gripper),
            self._move_to(
                arm, world, place_point - self.approach_distance * self.pointing
            ),
        ]

    def _move_to(
        self,
        arm: Arm,
        world: World,
        grip_point: NDArray[numpy.float64],
        position_threshold: float = MoveManipulatorAction.position_threshold,
    ) -> MoveManipulatorAction:
        """
        :param arm: The arm to move.
        :param world: The world.
        :param grip_point: Where its gripper is to grip.
        :param position_threshold: How close, in metres, the tool frame has to come.
        :return: The step moving the gripper there, pointing along :attr:`pointing`.
        """
        return MoveManipulatorAction(
            arm=arm,
            goal_pose=self.tool_pose(world, grip_point, self.pointing, self.closing),
            position_threshold=position_threshold,
        )

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
