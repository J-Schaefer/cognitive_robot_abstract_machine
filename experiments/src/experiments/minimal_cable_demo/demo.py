"""
DAiSy picks a cable up from its table with its right gripper, hands it to its left
gripper and drops it back onto the table; both hold it by friction alone. The cable is
simulated by SuperDex and the robot by MuJoCo; the switches of
:class:`~experiments.minimal_cable_demo.daisy_cable_scene.CommandLineFlag` steer a run::

    python3 -m experiments.minimal_cable_demo.demo --headless --video cable.mp4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
)
from cramph.node import StatechartNode
from experiments.minimal_cable_demo.daisy_cable_scene import (
    DAiSyCableDemonstration,
)
from numpy.typing import NDArray
from semantic_digital_twin.adapters.superdex.mujoco_cable_simulation import (
    MujocoSuperdexSim,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.deformable_cable import DeformableCable

# %% outcome


@dataclass(kw_only=True)
class HandoverOutcome:
    """
    Where the cable was while the left gripper alone held it, and where it ended up.
    """

    lowest_node_height_above_table_after_handover: float
    """
    How far the lowest node of the cable was above the table once the right gripper had
    let go, in metres; a cable the left gripper failed to take lies on the table, its
    lowest node one radius above it.
    """

    highest_node_height_above_table_after_drop: float
    """
    How far the highest node of the cable is above the table once the left gripper
    dropped it, in metres; a cable lying flat on the table has every node one radius
    above it.
    """


# %% demonstration


@dataclass(kw_only=True)
class MinimalCableDemonstration(DAiSyCableDemonstration):
    """
    DAiSy picks a cable up from its table with its right gripper, hands it to its left
    one and drops it back onto the table.
    """

    hold_point: tuple[float, float, float] = (0.35, 0.55, 0.45)
    """
    Where the right gripper holds the cable up, in the frame of the world root but with
    its height measured from the table's top, in metres.
    """

    take_arc_length: float = 0.15
    """
    How far down the cable from the right gripper's grip the left gripper takes it, in
    metres.
    """

    drop_height: float = 0.15
    """
    How far above the table's top the left gripper lowers the cable before it lets go of
    it, in metres.
    """

    left_pointing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([0.0, -1.0, 0.0])
    )
    """
    The direction the left gripper points along while it holds the cable.
    """

    left_closing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([-1.0, 0.0, 0.0])
    )
    """
    The direction the left gripper's fingers close along while it holds the cable.
    """

    settle_duration: timedelta = timedelta(seconds=1)
    """
    How long the cable is left to come to rest on the table, after the right gripper
    lifted it and after it was dropped.
    """

    def run(self) -> HandoverOutcome:
        """
        Set the scene up, pick the cable up, hand it over, drop it and measure where it
        was.

        :return: Where the cable was.
        """
        world, daisy = self.create_world()
        self.perform_kinematically(
            daisy=daisy, steps=[ParkArmsAction(arms=daisy.all_arms)]
        )
        cable = self.lay_cable_on_the_table(world=world)
        simulation = self.create_simulation(world=world)
        try:
            self.let_the_cable_settle(
                simulation=simulation, duration=self.settle_duration
            )
            self.perform_physically(
                daisy=daisy,
                simulation=simulation,
                steps=self.hold_up(
                    daisy=daisy, pick_point=self.pick_point(cable=cable)
                ),
            )
            self.let_the_cable_settle(
                simulation=simulation, duration=self.settle_duration
            )
            self.perform_physically(
                daisy=daisy,
                simulation=simulation,
                steps=[
                    self.approach(daisy=daisy, take_point=self.take_point(cable=cable))
                ],
            )
            # The approach may have brushed the cable aside.
            self.perform_physically(
                daisy=daisy,
                simulation=simulation,
                steps=self.take_over(
                    daisy=daisy,
                    simulation=simulation,
                    take_point=self.take_point(cable=cable),
                ),
            )
            height_after_handover = float(
                cable.centreline.corners[:, 2].min() - self.table_top(world=world)
            )
            self.perform_physically(
                daisy=daisy,
                simulation=simulation,
                steps=self.drop(daisy=daisy, simulation=simulation),
            )
            self.let_the_cable_settle(
                simulation=simulation, duration=self.settle_duration
            )
            return HandoverOutcome(
                lowest_node_height_above_table_after_handover=height_after_handover,
                highest_node_height_above_table_after_drop=float(
                    cable.centreline.corners[:, 2].max() - self.table_top(world=world)
                ),
            )
        finally:
            self.tear_down(simulation=simulation)

    def hold_up(
        self, *, daisy: DAiSy, pick_point: NDArray[numpy.float64]
    ) -> list[StatechartNode]:
        """
        :param daisy: The robot.
        :param pick_point: Where the right gripper grips the cable.
        :return: The steps in which the right gripper picks the cable up, lifts it
            straight up and carries it to :attr:`hold_point`.
        """
        arm = daisy.right_arm
        hold_point = numpy.array(self.hold_point) + numpy.array(
            [0.0, 0.0, self.table_top(world=daisy._world)]
        )
        return [
            *self.pick_up(arm=arm, pick_point=pick_point),
            self.move_from_above(
                arm=arm, grip_point=numpy.array([*pick_point[:2], hold_point[2]])
            ),
            self.move_from_above(arm=arm, grip_point=hold_point),
        ]

    def take_point(self, *, cable: DeformableCable) -> NDArray[numpy.float64]:
        """
        :param cable: The cable, hanging from the right gripper.
        :return: Where on the cable the left gripper takes it, as the cable lies now.
        """
        return cable.point_at_arc_length(
            arc_length=cable.centreline.length
            - self.pick_arc_length
            - self.take_arc_length
        ).to_np()[:3]

    def approach(
        self, *, daisy: DAiSy, take_point: NDArray[numpy.float64]
    ) -> MoveManipulatorAction:
        """
        :param daisy: The robot.
        :param take_point: Where on the cable the left gripper takes it.
        :return: The step bringing the left gripper :attr:`approach_distance` before
            the take point.
        """
        return self.move_from_the_side(
            arm=daisy.left_arm,
            grip_point=take_point - self.approach_distance * self.left_pointing,
        )

    def take_over(
        self,
        *,
        daisy: DAiSy,
        simulation: MujocoSuperdexSim,
        take_point: NDArray[numpy.float64],
    ) -> list[StatechartNode]:
        """
        :param daisy: The robot, its left gripper before the take point.
        :param simulation: The running simulation.
        :param take_point: Where on the cable the left gripper takes it.
        :return: The steps in which the left gripper takes the cable, and the right
            gripper lets go of it and backs off upwards.
        """
        right_arm = daisy.right_arm
        right_grip_point = self.measured_grip_point(
            simulation=simulation, gripper=right_arm.end_effector
        )
        return [
            self.move_from_the_side(
                arm=daisy.left_arm,
                grip_point=take_point,
                position_threshold=self.approach_position_threshold,
            ),
            self.close_on_cable(gripper=daisy.left_arm.end_effector),
            self.open_gripper(arm=right_arm),
            self.move_from_above(
                arm=right_arm,
                grip_point=right_grip_point - self.approach_distance * self.pointing,
            ),
        ]

    def drop(
        self, *, daisy: DAiSy, simulation: MujocoSuperdexSim
    ) -> list[StatechartNode]:
        """
        :param daisy: The robot, its left gripper holding the cable.
        :param simulation: The running simulation.
        :return: The steps in which the left gripper lowers the cable straight down
            to :attr:`drop_height`, lets go of it and backs off.
        """
        arm = daisy.left_arm
        drop_point = self.measured_grip_point(
            simulation=simulation, gripper=arm.end_effector
        )
        drop_point[2] = self.table_top(world=daisy._world) + self.drop_height
        return [
            self.move_from_the_side(arm=arm, grip_point=drop_point),
            self.open_gripper(arm=arm),
            self.move_from_the_side(
                arm=arm,
                grip_point=drop_point - self.approach_distance * self.left_pointing,
            ),
        ]

    def move_from_the_side(
        self,
        *,
        arm: Arm,
        grip_point: NDArray[numpy.float64],
        position_threshold: float = MoveManipulatorAction.position_threshold,
    ) -> MoveManipulatorAction:
        """
        :param arm: The arm to move.
        :param grip_point: Where its gripper is to grip.
        :param position_threshold: How close, in metres, the tool frame has to come.
        :return: The step moving the gripper there, pointing along
            :attr:`left_pointing`.
        """
        return MoveManipulatorAction(
            arm=arm,
            goal_pose=self.tool_pose(
                world=arm._world,
                grip_point=grip_point,
                pointing=self.left_pointing,
                closing=self.left_closing,
            ),
            position_threshold=position_threshold,
        )


# %% command line


def run_from_command_line() -> None:
    """
    Parse the command line and run the demonstration.
    """
    print(
        MinimalCableDemonstration.from_command_line(
            description="DAiSy hands a cable from one gripper to the other."
        ).run()
    )


if __name__ == "__main__":
    run_from_command_line()
