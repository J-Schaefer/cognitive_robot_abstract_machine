"""
DAiSy hands a cable from one gripper to the other, with the cable simulated by SuperDex
and the robot by MuJoCo.

The right gripper holds the top end of a cable hanging above the table. The left gripper
takes the cable from the side further down, lifts it, and the right gripper lets go, so
the cable ends up hanging from the left gripper. Both grippers hold the cable by friction
alone.

Run it as a script; ``--headless`` runs it without the MuJoCo viewer::

    python3 -m experiments.minimal_cable_demo.demo

It needs the ``superdex`` extra of ``semantic_digital_twin`` and the DAiSy description.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

import numpy
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import (
    PhysicallySimulatedRobot,
    kinematically_simulated_robot,
)
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveGripperMotion, MoveToolCenterPointMotion
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from semantic_digital_twin.adapters.superdex.mujoco_cable_simulation import (
    MujocoBodyPoses,
    MujocoSuperdexSim,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyGripper
from semantic_digital_twin.robots.griplink_gripper import GriplinkFlexConfiguration
from semantic_digital_twin.semantic_annotations.deformable_cable import DeformableCable
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color

# %% names in the DAiSy description


class DAiSyCableBody(StrEnum):
    """
    The bodies of DAiSy the cable can touch, as its description names them.
    """

    TABLE = "table_center"
    LEFT_GRIPPER_LEFT_FINGER_TIP = "left_gripper_left_finger_tip_link"
    LEFT_GRIPPER_RIGHT_FINGER_TIP = "left_gripper_right_finger_tip_link"
    RIGHT_GRIPPER_LEFT_FINGER_TIP = "right_gripper_left_finger_tip_link"
    RIGHT_GRIPPER_RIGHT_FINGER_TIP = "right_gripper_right_finger_tip_link"


class CommandLineFlag(StrEnum):
    """
    The switches of the demonstration's command line.
    """

    HEADLESS = "--headless"
    """
    Run without the MuJoCo viewer.
    """


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
class MinimalCableDemonstration:
    """
    DAiSy hands a hanging cable from its right gripper to its left one.
    """

    headless: bool = False
    """
    Whether to run without the MuJoCo viewer.
    """

    cable_length: float = 0.4
    """
    The length of the cable, in metres.
    """

    cable_radius: float = 0.0025
    """
    The radius of the cable, in metres.
    """

    number_of_cable_segments: int = 20
    """
    How many segments the cable is simulated with.
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

    grip_depth: float = 0.012
    """
    How far beyond the tool frame, along the approach axis, the ridge of DAiSy's finger
    tips grips a cable that crosses the fingers, in metres.
    """

    cable_overhang: float = 0.012
    """
    How far the top end of the cable reaches past the grip point of the right gripper,
    in metres: half the width of a finger tip, so the cable crosses the whole tip.
    """

    cable_grip_position: float = 0.0024
    """
    The opening width the grippers close to on the cable, in metres. It leaves 1.8 mm
    between the cable's axis and each ridge of the finger tips, 0.7 mm less than the
    cable's radius.
    """

    grip_position_threshold: float = 1.0e-4
    """
    How close, in metres, the fingers have to come to the commanded width before a
    grip counts as done.
    """

    approach_position_threshold: float = 1.0e-3
    """
    How close, in metres, the left tool frame has to come to the point it grips the
    cable from.
    """

    control_period: timedelta = timedelta(milliseconds=20)
    """
    The period the physics is stepped by while the cable settles.
    """

    settle_duration: timedelta = timedelta(seconds=1)
    """
    How long the cable hangs pinned to the right gripper before the gripper's friction
    alone holds it.
    """

    cable_color: Color = field(default_factory=Color.WHITE)
    """
    The colour the cable is drawn in.
    """

    def run(self) -> HandoverOutcome:
        """
        Set the scene up, hand the cable over and measure where it ended up.

        :return: Where the cable ended up.
        """
        world = URDFParser.from_file(file_path=DAiSy.get_ros_file_path()).parse()
        daisy = DAiSy.from_world(world)
        context = Context(world, daisy)
        self.hold_cable_with_the_right_gripper(context)
        cable = self.hang_cable_from_the_right_gripper(world, daisy)
        simulation = MujocoSuperdexSim(
            world=world,
            colliding_bodies=[world.get_body_by_name(name) for name in DAiSyCableBody],
            headless=self.headless,
        )
        simulation.start_stepped_simulation()
        try:
            self.settle(simulation, cable, daisy)
            take_point = cable.point_at_arc_length(self.take_arc_length).to_np()[:3]
            with PhysicallySimulatedRobot(simulation=simulation):
                self.hand_over(context, take_point).perform()
            return self.outcome(simulation, cable, daisy, world)
        finally:
            simulation.stop_simulation()

    def hold_cable_with_the_right_gripper(self, context: Context) -> None:
        """
        Move the right gripper, kinematically, to where it holds the cable: pointing
        along the world's y axis and closed to the cable's width.

        :param context: The plan context of the robot.
        """
        right_arm = context.robot.right_arm
        hold_point = numpy.array(self.hold_point)
        plan = sequential(
            [
                ParkArmsAction(context.robot.all_arms),
                MoveToolCenterPointMotion(
                    target=self._tool_pose(
                        context.world,
                        hold_point,
                        pointing=numpy.array([0.0, 1.0, 0.0]),
                        closing=numpy.array([1.0, 0.0, 0.0]),
                    ),
                    arm=right_arm,
                ),
                self._close_on_cable(right_arm.end_effector),
            ],
            context,
        ).plan
        with kinematically_simulated_robot:
            plan.perform()

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
        top = self._grip_point(
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
        for _ in range(round(self.settle_duration / self.control_period)):
            simulation.step_simulation(self.control_period)
        simulation.cable_simulation.release(pin)

    def hand_over(self, context: Context, take_point: NDArray[numpy.float64]) -> Plan:
        """
        :param context: The plan context of the robot.
        :param take_point: Where on the cable the left gripper takes it.
        :return: The plan handing the cable from the right gripper to the left one.
        """
        left_arm = context.robot.left_arm
        pointing = numpy.array([0.0, -1.0, 0.0])
        closing = numpy.array([-1.0, 0.0, 0.0])
        world = context.world
        return sequential(
            [
                MoveToolCenterPointMotion(
                    target=self._tool_pose(
                        world,
                        take_point - self.approach_distance * pointing,
                        pointing,
                        closing,
                    ),
                    arm=left_arm,
                ),
                MoveToolCenterPointMotion(
                    target=self._tool_pose(world, take_point, pointing, closing),
                    arm=left_arm,
                    position_threshold=self.approach_position_threshold,
                ),
                self._close_on_cable(left_arm.end_effector),
                MoveToolCenterPointMotion(
                    target=self._tool_pose(
                        world, take_point + numpy.array(self.lift), pointing, closing
                    ),
                    arm=left_arm,
                ),
                MoveGripperMotion(
                    configuration=context.robot.right_arm.end_effector.default_configuration(
                        GripperState.OPEN
                    )
                ),
            ],
            context,
        ).plan

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
        table = world.get_body_by_name(DAiSyCableBody.TABLE)
        table_top = (
            table.collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
            .max_z
        )
        left_grip_point = self._grip_point(
            MujocoBodyPoses(simulator=simulation.simulator).world_T_body(
                daisy.left_arm.end_effector.tool_frame
            )
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
                centreline.corners[:, 2].min() - table_top
            ),
            distance_from_left_grip_point=float(
                numpy.linalg.norm(points_along - left_grip_point, axis=1).min()
            ),
        )

    def _grip_point(
        self, world_T_tool: NDArray[numpy.float64]
    ) -> NDArray[numpy.float64]:
        """
        :param world_T_tool: The pose of a gripper's tool frame.
        :return: The point that gripper grips a cable crossing its fingers at.
        """
        return world_T_tool[:3, 3] + self.grip_depth * world_T_tool[:3, 2]

    def _tool_pose(
        self,
        world: World,
        grip_point: NDArray[numpy.float64],
        pointing: NDArray[numpy.float64],
        closing: NDArray[numpy.float64],
    ) -> Pose:
        """
        :param world: The world.
        :param grip_point: Where the gripper is to grip, in the frame of the world root.
        :param pointing: The direction the gripper is to point along.
        :param closing: The direction its fingers are to close along.
        :return: The pose of the tool frame that grips there.
        """
        rotation = numpy.stack([closing, numpy.cross(pointing, closing), pointing], 1)
        position = grip_point - self.grip_depth * pointing
        x, y, z, w = Rotation.from_matrix(rotation).as_quat()
        return Pose.from_xyz_quaternion(
            *position, x, y, z, w, reference_frame=world.root
        )

    def _close_on_cable(self, gripper: DAiSyGripper) -> MoveGripperMotion:
        """
        :param gripper: A gripper of the robot.
        :return: The motion closing it to the cable's width.
        """
        configuration = GriplinkFlexConfiguration.from_state_type(
            gripper, GripperState.FLEXCLOSE, grip_position=self.cable_grip_position
        )
        configuration.position_threshold = self.grip_position_threshold
        return MoveGripperMotion(configuration=configuration)


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
    arguments = parser.parse_args()
    outcome = MinimalCableDemonstration(headless=arguments.headless).run()
    print(outcome)


if __name__ == "__main__":
    run_from_command_line()
