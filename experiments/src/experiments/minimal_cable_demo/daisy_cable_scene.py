"""
What the cable demonstrations with DAiSy share: the robot's world, the poses and finger
positions that grip a cable, and running a plan kinematically or against the physics,
with the cable simulated by SuperDex, the robot by MuJoCo and its motions controlled by
a Giskard in this process or by a Giskard node already running.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy
from coraplex.demonstrations import (
    WORLD_FETCH_TIMEOUT_SECONDS,
    RobotDemonstrationRosSession,
)
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PhysicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.executor import SteppedSimulationPacer
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood.patterns.called_positionally import called_positionally
from krrood_spatial_math.spatial_types import Point3, Pose
from numpy.typing import NDArray
from physics_simulators.mujoco_simulator import (
    HeadlessGraphicsBackend,
    MujocoEnvironmentVariable,
)
from scipy.spatial.transform import Rotation
from semantic_digital_twin.adapters.ros.world_fetcher import join_shared_world
from semantic_digital_twin.adapters.superdex.mujoco_cable_simulation import (
    MujocoBodyPoses,
    MujocoSuperdexSim,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyGripper
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.deformable_cable import DeformableCable
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import Any, Self

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
    The switches of the demonstrations' command lines.
    """

    HEADLESS = "--headless"
    """
    Run without the MuJoCo viewer.
    """

    VIDEO = "--video"
    """
    Record a video of the run into the file that follows.
    """

    RUNNING_GISKARD = "--running-giskard"
    """
    Send the robot's motions to a Giskard node already running.
    """


# %% video


@dataclass(kw_only=True)
class CableVideo:
    """
    Frames of a simulation with its cables, rendered offscreen from a fixed camera.
    """

    path: Path
    """
    The file the video is written to.
    """

    frames_per_second: int = 30
    """
    The rate the video plays back at, in simulated time.
    """

    width: int = 960
    """
    The width of a frame, in pixels.
    """

    height: int = 720
    """
    The height of a frame, in pixels.
    """

    look_at: tuple[float, float, float] = (0.4, 0.6, 0.75)
    """
    The point the camera looks at, in the frame of the world root, in metres.
    """

    distance: float = 1.4
    """
    How far the camera is from :attr:`look_at`, in metres.
    """

    azimuth: float = 200.0
    """
    The direction the camera looks along, about the vertical axis, in degrees.
    """

    elevation: float = -40.0
    """
    How far the camera looks down, in degrees.
    """

    frames: list[NDArray[numpy.uint8]] = field(init=False, default_factory=list)
    """
    The frames captured so far.
    """

    _renderer: Any = field(init=False, default=None)
    """
    The offscreen renderer, built with the first frame.
    """

    _simulated_time_since_frame: timedelta = field(
        init=False, default_factory=timedelta
    )
    """
    How much simulated time passed since the last frame was captured.
    """

    def __post_init__(self) -> None:
        """
        Select a headless OpenGL backend for the offscreen rendering, unless one is
        selected already.
        """
        os.environ.setdefault(
            MujocoEnvironmentVariable.GRAPHICS_BACKEND, HeadlessGraphicsBackend.EGL
        )

    def advance(self, *, simulation: MujocoSuperdexSim, duration: timedelta) -> None:
        """
        Note that the simulation advanced, capturing a frame whenever a frame period
        has passed.

        :param simulation: The simulation, standing where the advance left it.
        :param duration: How much simulated time passed.
        """
        self._simulated_time_since_frame += duration
        frame_period = timedelta(seconds=1 / self.frames_per_second)
        if self._simulated_time_since_frame < frame_period:
            return
        self._simulated_time_since_frame -= frame_period
        self.frames.append(self._render(simulation=simulation))

    def _render(self, *, simulation: MujocoSuperdexSim) -> NDArray[numpy.uint8]:
        """
        :param simulation: The simulation to render.
        :return: The frame showing it as it stands, with its cables.
        """
        model = simulation.simulator._mj_model
        data = simulation.simulator._mj_data
        if self._renderer is None:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
            self._renderer = mujoco.Renderer(model, self.height, self.width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.look_at
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
        self._renderer.update_scene(data, camera=camera)
        simulation.draw_cables(scene=self._renderer.scene)
        return self._renderer.render()

    def write(self) -> None:
        """
        Encode the frames captured so far into :attr:`path`.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(self.path, self.frames, fps=self.frames_per_second)


class RecordedMujocoSuperdexSim(MujocoSuperdexSim):
    """
    A :class:`MujocoSuperdexSim` that captures a video while it is stepped.
    """

    video: CableVideo
    """
    The video the steps are captured into.
    """

    def __init__(self, *, video: CableVideo, **kwargs: Any) -> None:
        """
        :param video: The video to capture the steps into.
        :param kwargs: See :class:`MujocoSuperdexSim`.
        """
        super().__init__(**kwargs)
        self.video = video

    @called_positionally
    def step_simulation(self, duration: timedelta) -> None:
        """
        Advance the simulation as :meth:`MujocoSuperdexSim.step_simulation` does, and
        capture it into :attr:`video`.
        """
        super().step_simulation(duration)
        self.video.advance(simulation=self, duration=duration)


# %% following a running Giskard


@dataclass(kw_only=True)
class CommandedJointPositions:
    """
    The joint positions a running Giskard commanded at one moment.
    """

    time: float
    """
    How long after the start of the step it was commanded in, in seconds.
    """

    positions: dict[PrefixedName, float]
    """
    The commanded positions, by the name of their degree of freedom.
    """

    @classmethod
    def from_world(cls, *, world: World, start: float) -> Self:
        """
        :param world: The world the running Giskard serves.
        :param start: The moment the step started at, on the monotonic clock, in
            seconds.
        :return: The positions the world holds now.
        """
        return cls(
            time=time.monotonic() - start, positions=world.state.to_position_dict()
        )


# %% demonstration


@dataclass(kw_only=True)
class DAiSyCableDemonstration:
    """
    A demonstration in which DAiSy handles a cable.
    """

    headless: bool = False
    """
    Whether to run without the MuJoCo viewer.
    """

    video: CableVideo | None = None
    """
    The video to record the run into, if any.
    """

    uses_running_giskard: bool = False
    """
    Whether the robot's motions are sent to a Giskard node already running, whose world
    the demonstration joins, instead of being controlled by a Giskard in this process.
    """

    ros_node_name: str = "daisy_cable_demonstration"
    """
    The name of the node that talks to the running Giskard.
    """

    ros_session: RobotDemonstrationRosSession | None = field(init=False, default=None)
    """
    The ROS session the running Giskard is talked to through, ``None`` until
    :meth:`create_world` joins its world.
    """

    giskard_world: World | None = field(init=False, default=None)
    """
    The world the running Giskard serves, kept in step with it, ``None`` until
    :meth:`create_world` joins it.
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

    grip_depth: float = 0.02
    """
    How far beyond the tool frame, along the approach axis, the ridges of DAiSy's
    fingertips grip a cable that crosses the fingers, in metres: about the middle of the
    ridges, which reach from 8 mm to 31 mm beyond the tool frame.
    """

    finger_position_on_cable: float = 0.041
    """
    The position of a gripper's finger joint when it closes on the cable, in metres. It
    leaves 1.8 mm between the cable's axis and each ridge of the fingertips, 0.7 mm less
    than the cable's radius; the ridges stand 85.6 mm apart with the gripper open and
    each finger travels as far as its joint.
    """

    grip_position_threshold: float = 1.0e-4
    """
    How close, in metres, the fingers have to come to
    :attr:`finger_position_on_cable` before a grip counts as done.
    """

    finger_closing_velocity: float = 0.02
    """
    How fast the fingers close on the cable, in metres per second.
    """

    control_period: timedelta = timedelta(milliseconds=20)
    """
    The period the physics is stepped by while the cable settles, and while a running
    Giskard moves the robot.
    """

    cable_color: Color = field(default_factory=Color.ORANGE)
    """
    The colour the cable is drawn in.
    """

    # Past the edge, since the fingertips reach 11 mm beyond the groove that holds the
    # cable and would hit the table before gripping a cable lying flat on it.
    cable_end: tuple[float, float] = (0.66, 0.45)
    """
    Where the end of the cable lies, 6 cm past the table's far edge, in the frame of
    the world root, in metres; the cable lies from there along the world's negative x
    axis.
    """

    pick_arc_length: float = 0.025
    """
    How far from its end a gripper grips the cable lying on the table, in metres: far
    enough past the table's edge that the fingertips pass it.
    """

    approach_distance: float = 0.1
    """
    How far from the cable a gripper stops before it closes in on it, and how far it
    backs off once it let go, in metres.
    """

    approach_position_threshold: float = 1.0e-3
    """
    How close, in metres, a tool frame has to come to where it grips the cable.
    """

    pointing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([0.0, 0.0, -1.0])
    )
    """
    The direction a gripper points along when it grips the cable from above: straight
    down.
    """

    closing: NDArray[numpy.float64] = field(
        default_factory=lambda: numpy.array([0.0, 1.0, 0.0])
    )
    """
    The direction the fingers of a gripper gripping from above close along: across the
    cable lying on the table.
    """

    @classmethod
    def from_command_line(cls, *, description: str) -> Self:
        """
        :param description: What the demonstration does, for the command line's help.
        :return: The demonstration the command line's switches ask for.
        """
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            CommandLineFlag.HEADLESS,
            action="store_true",
            help="Run without the MuJoCo viewer.",
        )
        parser.add_argument(
            CommandLineFlag.VIDEO, type=Path, help="Record a video into this file."
        )
        parser.add_argument(
            CommandLineFlag.RUNNING_GISKARD,
            action="store_true",
            help="Send the robot's motions to a Giskard node already running.",
        )
        arguments = parser.parse_args()
        return cls(
            headless=arguments.headless,
            video=None if arguments.video is None else CableVideo(path=arguments.video),
            uses_running_giskard=arguments.running_giskard,
        )

    def create_world(self) -> tuple[World, DAiSy]:
        """
        :return: A new world holding DAiSy, to add the cable to and simulate, and the
            DAiSy that performs the plans: the one in that world, or the one in
            :attr:`giskard_world` if :attr:`uses_running_giskard`.
        """
        world = URDFParser.from_file(file_path=DAiSy.get_ros_file_path()).create_world()
        simulated_daisy = DAiSy.from_world(world)
        if not self.uses_running_giskard:
            return world, simulated_daisy
        self.ros_session = RobotDemonstrationRosSession.start(self.ros_node_name)
        self.giskard_world = join_shared_world(
            self.ros_session.node, timeout_seconds=WORLD_FETCH_TIMEOUT_SECONDS
        )
        [giskard_daisy] = self.giskard_world.get_semantic_annotations_by_type(DAiSy)
        return world, giskard_daisy

    def create_simulation(self, *, world: World) -> MujocoSuperdexSim:
        """
        :param world: The world holding DAiSy and the cable.
        :return: A new, started simulation of the world in which the cable touches the
            table and the fingertips, recording :attr:`video` if there is one.
        """
        if self.uses_running_giskard:
            self._command_joint_positions(
                world=world, positions=self.giskard_world.state.to_position_dict()
            )
        colliding_bodies = [world.get_body_by_name(name) for name in DAiSyCableBody]
        if self.video is None:
            simulation = MujocoSuperdexSim(
                world=world, colliding_bodies=colliding_bodies, headless=self.headless
            )
        else:
            simulation = RecordedMujocoSuperdexSim(
                video=self.video,
                world=world,
                colliding_bodies=colliding_bodies,
                headless=self.headless,
            )
        simulation.start_stepped_simulation()
        if self.uses_running_giskard:
            # The world holds the commands the servos follow, which the physics must
            # not overwrite.
            simulation.synchronizer.sync_rate_hz = 0
        return simulation

    def tear_down(self, *, simulation: MujocoSuperdexSim) -> None:
        """
        Stop the simulation, write :attr:`video` if there is one and leave the ROS
        session if one was started.

        :param simulation: The running simulation.
        """
        simulation.stop_simulation()
        if self.video is not None:
            self.video.write()
        if self.ros_session is None:
            return
        self.ros_session.stop()
        self.ros_session = None

    def let_the_cable_settle(
        self, *, simulation: MujocoSuperdexSim, duration: timedelta
    ) -> None:
        """
        :param simulation: The running simulation.
        :param duration: How long to step the physics for.
        """
        for _ in range(round(duration / self.control_period)):
            simulation.step_simulation(self.control_period)

    def perform_kinematically(
        self, *, daisy: DAiSy, steps: list[StatechartNode]
    ) -> None:
        """
        Perform steps one after the other without the physics, with the world following
        every command exactly, or as the running Giskard moves the robot.

        :param daisy: The robot performing them.
        :param steps: The steps.
        """
        if self.uses_running_giskard:
            self._perform(
                executor=self._running_giskard_executor(daisy=daisy), steps=steps
            )
            return
        self._perform(
            executor=KinematicallySimulatedPlanExecutor(
                context=self._context(daisy=daisy)
            ),
            steps=steps,
        )

    def perform_physically(
        self,
        *,
        daisy: DAiSy,
        simulation: MujocoSuperdexSim,
        steps: list[StatechartNode],
    ) -> None:
        """
        Perform steps one after the other, against the physics.

        A running Giskard does not wait for the physics, so every step is sent to it on
        its own, see :meth:`_replay_running_giskard`, and the next one only once the
        simulated robot has come to a standstill.

        :param daisy: The robot performing them.
        :param simulation: The running simulation.
        :param steps: The steps.
        """
        if not self.uses_running_giskard:
            self._perform(
                executor=PhysicallySimulatedPlanExecutor(
                    context=self._context(daisy=daisy), simulation=simulation
                ),
                steps=steps,
            )
            return
        pacer = SteppedSimulationPacer(simulation=simulation)
        pacer.target_frequency = 1 / self.control_period.total_seconds()
        for step in steps:
            self._replay_running_giskard(daisy=daisy, simulation=simulation, step=step)
            pacer.wait_for_standstill()

    def _replay_running_giskard(
        self, *, daisy: DAiSy, simulation: MujocoSuperdexSim, step: StatechartNode
    ) -> None:
        """
        Send a step to the running Giskard and let the simulated robot follow the joint
        positions it commands, replayed at the pace they were commanded at: Giskard
        moves the robot in real time, which the physics may be too slow to keep.

        :param daisy: The robot performing the step, in :attr:`giskard_world`.
        :param simulation: The running simulation.
        :param step: The step.
        """
        executor = self._running_giskard_executor(daisy=daisy)
        self._compile(executor=executor, steps=[step])
        start = time.monotonic()
        commands = [
            CommandedJointPositions.from_world(world=self.giskard_world, start=start)
        ]
        simulated_time = 0.0
        end: float | None = None
        # The viewer and the video render on this thread, so the plan runs on another.
        with ThreadPoolExecutor(max_workers=1) as plan_thread:
            execution = plan_thread.submit(executor.execute)
            while end is None or simulated_time < end:
                if end is None:
                    finished = execution.done()
                    commands.append(
                        CommandedJointPositions.from_world(
                            world=self.giskard_world, start=start
                        )
                    )
                    if finished:
                        end = commands[-1].time
                simulated_time += self.control_period.total_seconds()
                command = next(
                    command
                    for command in reversed(commands)
                    if command.time <= simulated_time
                )
                self._command_joint_positions(
                    world=simulation.world, positions=command.positions
                )
                simulation.step_simulation(self.control_period)
            execution.result()
        executor.close()

    @staticmethod
    def _command_joint_positions(
        *, world: World, positions: dict[PrefixedName, float]
    ) -> None:
        """
        :param world: The world whose joints to command.
        :param positions: The positions to command them to, by the name of their degree
            of freedom.
        """
        for name, position in positions.items():
            world.state[
                world.get_degree_of_freedom_by_name(name).id
            ].position = position
        world.notify_state_change()

    def _running_giskard_executor(self, *, daisy: DAiSy) -> RealRobotPlanExecutor:
        """
        :param daisy: The robot.
        :return: A new executor sending a plan DAiSy performs to the running Giskard.
        """
        return RealRobotPlanExecutor(
            context=self._context(daisy=daisy), ros_node=self.ros_session.node
        )

    @staticmethod
    def _context(*, daisy: DAiSy) -> StatechartContext:
        """
        :param daisy: The robot.
        :return: A new context of a plan DAiSy performs.
        """
        return StatechartContext.with_extensions(
            world=daisy._world, extensions=[RobotAccess(robot=daisy)]
        )

    @staticmethod
    def _perform(*, executor: PlanExecutor, steps: list[StatechartNode]) -> None:
        """
        :param executor: The executor to perform the steps with.
        :param steps: The steps, performed one after the other.
        """
        DAiSyCableDemonstration._compile(executor=executor, steps=steps)
        executor.execute()
        executor.close()

    @staticmethod
    def _compile(*, executor: PlanExecutor, steps: list[StatechartNode]) -> None:
        """
        :param executor: The executor to perform the steps with.
        :param steps: The steps, to be performed one after the other.
        """
        statechart = Statechart(context=executor.context)
        statechart.add_node(node=Sequence(nodes=steps))
        executor.compile(statechart=statechart)

    def lay_cable_on_the_table(self, *, world: World) -> DeformableCable:
        """
        Add a straight cable to the world, lying on the table along the world's x axis
        with its end at :attr:`cable_end`, past the table's far edge.

        :param world: The world to add the cable to.
        :return: The cable.
        """
        end = numpy.array(
            [*self.cable_end, self.table_top(world=world) + self.cable_radius]
        )
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

    def pick_point(self, *, cable: DeformableCable) -> NDArray[numpy.float64]:
        """
        :param cable: The cable.
        :return: Where a gripper grips the cable from above, as the cable lies now.
        """
        return cable.point_at_arc_length(
            arc_length=cable.centreline.length - self.pick_arc_length
        ).to_np()[:3]

    def pick_up(
        self, *, arm: Arm, pick_point: NDArray[numpy.float64]
    ) -> list[StatechartNode]:
        """
        :param arm: The arm to pick the cable up with.
        :param pick_point: Where its gripper grips the cable.
        :return: The steps opening the gripper, bringing it down onto the pick point
            from :attr:`approach_distance` above and closing it on the cable.
        """
        return [
            self.open_gripper(arm=arm),
            self.move_from_above(
                arm=arm, grip_point=pick_point - self.approach_distance * self.pointing
            ),
            self.move_from_above(
                arm=arm,
                grip_point=pick_point,
                position_threshold=self.approach_position_threshold,
            ),
            self.close_on_cable(gripper=arm.end_effector),
        ]

    def move_from_above(
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
        :return: The step moving the gripper there, pointing along :attr:`pointing`.
        """
        return MoveManipulatorAction(
            arm=arm,
            goal_pose=self.tool_pose(
                world=arm._world,
                grip_point=grip_point,
                pointing=self.pointing,
                closing=self.closing,
            ),
            position_threshold=position_threshold,
        )

    @staticmethod
    def open_gripper(*, arm: Arm) -> SetGripperAction:
        """
        :param arm: The arm whose gripper to open.
        :return: The step opening it.
        """
        return SetGripperAction(
            gripper_configuration=arm.end_effector.default_configuration(
                GripperState.OPEN
            )
        )

    def grip_point(
        self, *, world_T_tool: NDArray[numpy.float64]
    ) -> NDArray[numpy.float64]:
        """
        :param world_T_tool: The pose of a gripper's tool frame.
        :return: The point that gripper grips a cable crossing its fingers at.
        """
        return world_T_tool[:3, 3] + self.grip_depth * world_T_tool[:3, 2]

    def tool_pose(
        self,
        *,
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

    def close_on_cable(self, *, gripper: DAiSyGripper) -> JointPositionList:
        """
        :param gripper: A gripper of the robot.
        :return: The task closing its fingers on the cable.
        """
        [finger_connection] = gripper.finger_connections
        return JointPositionList(
            goal_state=JointState(
                connections=[finger_connection],
                target_values=[self.finger_position_on_cable],
            ),
            threshold=self.grip_position_threshold,
            cruise_velocity=self.finger_closing_velocity,
        )

    def measured_grip_point(
        self, *, simulation: MujocoSuperdexSim, gripper: DAiSyGripper
    ) -> NDArray[numpy.float64]:
        """
        :param simulation: The simulation the gripper moves in.
        :param gripper: A gripper of the robot.
        :return: Where the gripper grips a cable as the physics has it now.
        """
        return self.grip_point(
            world_T_tool=MujocoBodyPoses(simulator=simulation.simulator).world_T_body(
                body=gripper.tool_frame
            )
        )

    @staticmethod
    def table_top(*, world: World) -> float:
        """
        :param world: The world holding DAiSy.
        :return: The height of the top of DAiSy's table, in metres.
        """
        table = world.get_body_by_name(DAiSyCableBody.TABLE)
        return (
            table.collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
            .max_z
        )
