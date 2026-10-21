"""
DAiSy hands a cable from one gripper to the other, with the cable simulated by SuperDex
and the robot by MuJoCo.

The right gripper holds the top end of a cable hanging above the table. The left gripper
takes the cable from the side further down, lifts it, and the right gripper lets go, so
the cable ends up hanging from the left gripper. Both grippers hold the cable by friction
alone.

Run it as a script; ``--headless`` runs it without the MuJoCo viewer and ``--video``
records it::

    python3 -m experiments.minimal_cable_demo.demo

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
from semantic_digital_twin.adapters.superdex.mujoco_cable_simulation import (
    MujocoSuperdexSim,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.deformable_cable import DeformableCable
from semantic_digital_twin.world import World

# %% outcome


@dataclass
class HandoverOutcome:
    """
    Where the cable ended up after the handover.
    """

    lowest_node_height_above_table: float
    """
    How far the lowest node of the cable is above the table, in metres; a cable that
    was dropped lies on the table, its lowest node one radius above it.
    """

    distance_from_left_grip_point: float
    """
    How far the cable passes from the point the left gripper grips at, in metres.
    """


# %% demonstration


@dataclass
class MinimalCableDemonstration(DAiSyCableDemonstration):
    """
    DAiSy hands a hanging cable from its right gripper to its left one.
    """

    hold_point: tuple[float, float, float] = (0.35, 0.6, 1.0577)
    """
    Where the right gripper holds the cable, in the frame of the world root, in metres.
    """

    take_arc_length: float = 0.25
    """
    How far down the cable from its top the left gripper takes it, in metres.
    """

    approach_distance: float = 0.1
    """
    How far from the cable the left gripper stops before it closes in, in metres.
    """

    lift: tuple[float, float, float] = (0.0, 0.045, 0.15)
    """
    How far the left gripper moves the cable after taking it, in the frame of the world
    root, in metres.
    """

    left_pointing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([0.0, -1.0, 0.0])
    )
    """
    The direction the left gripper points along while it takes the cable.
    """

    left_closing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([-1.0, 0.0, 0.0])
    )
    """
    The direction the left gripper's fingers close along while it takes the cable.
    """

    cable_overhang: float = 0.012
    """
    How far the top end of the cable reaches past the grip point of the right gripper,
    in metres: half the width of a finger tip, so the cable crosses the whole tip.
    """

    approach_position_threshold: float = 1.0e-3
    """
    How close, in metres, the left tool frame has to come to the point it grips the
    cable from.
    """

    settle_duration: timedelta = timedelta(seconds=1)
    """
    How long the cable hangs pinned to the right gripper before the gripper's friction
    alone holds it.
    """

    def run(self) -> HandoverOutcome:
        """
        Set the scene up, hand the cable over and measure where it ended up.

        :return: Where the cable ended up.
        """
        world, daisy = self.create_world()
        self.hold_cable_with_the_right_gripper(daisy)
        cable = self.hang_cable_from_the_right_gripper(world, daisy)
        simulation = self.create_simulation(world)
        try:
            self.settle(simulation, cable, daisy)
            self.perform_physically(
                daisy, simulation, [self.approach(daisy, self.take_point(cable))]
            )
            # The approach may have brushed the cable aside.
            self.perform_physically(
                daisy, simulation, self.take_over(daisy, self.take_point(cable))
            )
            return self.outcome(simulation, cable, daisy, world)
        finally:
            simulation.stop_simulation()
            if self.video is not None:
                self.video.write()

    def hold_cable_with_the_right_gripper(self, daisy: DAiSy) -> None:
        """
        Move the right gripper, kinematically, to where it holds the cable: pointing
        along the world's y axis and closed to the cable's width.

        :param daisy: The robot.
        """
        right_arm = daisy.right_arm
        self.perform_kinematically(
            daisy,
            [
                ParkArmsAction(arms=daisy.all_arms),
                MoveManipulatorAction(
                    arm=right_arm,
                    goal_pose=self.tool_pose(
                        daisy._world,
                        numpy.array(self.hold_point),
                        pointing=numpy.array([0.0, 1.0, 0.0]),
                        closing=numpy.array([1.0, 0.0, 0.0]),
                    ),
                ),
                self.close_on_cable(right_arm.end_effector),
            ],
        )

    def hang_cable_from_the_right_gripper(
        self, world: World, daisy: DAiSy
    ) -> DeformableCable:
        """
        Add a straight cable to the world, hanging down from between the right
        gripper's fingers.

        :param world: The world to add the cable to.
        :param daisy: The robot.
        :return: The cable.
        """
        top = self.grip_point(
            world.compute_forward_kinematics_np(
                world.root, daisy.right_arm.end_effector.tool_frame
            )
        ) + numpy.array([0.0, 0.0, self.cable_overhang])
        bottom = top - numpy.array([0.0, 0.0, self.cable_length])
        cable = DeformableCable.along_polyline(
            name=PrefixedName("cable"),
            polyline=[
                Point3(*top, reference_frame=world.root),
                Point3(*bottom, reference_frame=world.root),
            ],
            number_of_segments=self.number_of_cable_segments,
            radius=self.cable_radius,
            color=self.cable_color,
        )
        with world.modify_world():
            world.add_semantic_annotation(cable)
        return cable

    def settle(
        self, simulation: MujocoSuperdexSim, cable: DeformableCable, daisy: DAiSy
    ) -> None:
        """
        Hold the top of the cable pinned to the right gripper while the simulated
        fingers close on it, then leave it to their friction.

        :param simulation: The running simulation.
        :param cable: The cable.
        :param daisy: The robot.
        """
        pin = simulation.cable_simulation.pin(
            cable, 0, daisy.right_arm.end_effector.tool_frame
        )
        self.let_the_cable_settle(simulation, self.settle_duration)
        simulation.cable_simulation.release(pin)

    def take_point(self, cable: DeformableCable) -> NDArray[numpy.float64]:
        """
        :param cable: The cable.
        :return: Where on the cable the left gripper takes it, as the cable lies now.
        """
        return cable.point_at_arc_length(self.take_arc_length).to_np()[:3]

    def approach(
        self, daisy: DAiSy, take_point: NDArray[numpy.float64]
    ) -> MoveManipulatorAction:
        """
        :param daisy: The robot.
        :param take_point: Where on the cable the left gripper takes it.
        :return: The step bringing the left gripper :attr:`approach_distance` before
            the take point.
        """
        return MoveManipulatorAction(
            arm=daisy.left_arm,
            goal_pose=self.tool_pose(
                daisy._world,
                take_point - self.approach_distance * self.left_pointing,
                self.left_pointing,
                self.left_closing,
            ),
        )

    def take_over(
        self, daisy: DAiSy, take_point: NDArray[numpy.float64]
    ) -> list[StatechartNode]:
        """
        :param daisy: The robot, its left gripper before the take point.
        :param take_point: Where on the cable the left gripper takes it.
        :return: The steps in which the left gripper takes the cable and lifts it, and
            the right gripper lets go of it.
        """
        left_arm = daisy.left_arm
        world = daisy._world
        return [
            MoveManipulatorAction(
                arm=left_arm,
                goal_pose=self.tool_pose(
                    world, take_point, self.left_pointing, self.left_closing
                ),
                position_threshold=self.approach_position_threshold,
            ),
            self.close_on_cable(left_arm.end_effector),
            MoveManipulatorAction(
                arm=left_arm,
                goal_pose=self.tool_pose(
                    world,
                    take_point + numpy.array(self.lift),
                    self.left_pointing,
                    self.left_closing,
                ),
            ),
            SetGripperAction(
                gripper_configuration=daisy.right_arm.end_effector.default_configuration(
                    GripperState.OPEN
                )
            ),
        ]

    def outcome(
        self,
        simulation: MujocoSuperdexSim,
        cable: DeformableCable,
        daisy: DAiSy,
        world: World,
    ) -> HandoverOutcome:
        """
        :param simulation: The simulation the handover ran in.
        :param cable: The cable.
        :param daisy: The robot.
        :param world: The world.
        :return: Where the cable ended up.
        """
        left_grip_point = self.measured_grip_point(
            simulation, daisy.left_arm.end_effector
        )
        centreline = cable.centreline
        points_along = numpy.array(
            [
                centreline.point_at(arc_length)
                for arc_length in numpy.linspace(0.0, centreline.length, 1000)
            ]
        )
        return HandoverOutcome(
            lowest_node_height_above_table=float(
                centreline.corners[:, 2].min() - self.table_top(world)
            ),
            distance_from_left_grip_point=float(
                numpy.linalg.norm(points_along - left_grip_point, axis=1).min()
            ),
        )


def run_from_command_line() -> None:
    """
    Parse the command line and run the demonstration.
    """
    parser = argparse.ArgumentParser(description="DAiSy hands a cable over.")
    parser.add_argument(
        CommandLineFlag.HEADLESS,
        action="store_true",
        help="Run without the MuJoCo viewer.",
    )
    parser.add_argument(
        CommandLineFlag.VIDEO, type=Path, help="Record a video into this file."
    )
    arguments = parser.parse_args()
    outcome = MinimalCableDemonstration(
        headless=arguments.headless,
        video=None if arguments.video is None else CableVideo(path=arguments.video),
    ).run()
    print(outcome)


if __name__ == "__main__":
    run_from_command_line()
