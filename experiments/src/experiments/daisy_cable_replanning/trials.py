from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from experiments.daisy_cable_replanning.cable_demo import DemoScript
from experiments.daisy_cable_replanning.collisions import (
    ActiveMotionTracker,
    ArmCollisionRecorder,
    IllegalJointStateDatabase,
)
from experiments.daisy_cable_replanning.replanning import (
    ReplanningLog,
    ReplanningParameters,
    ReplanningTransformation,
)
from krrood.adapters.json_serializer import to_json
from ompl import util as ompl_util
from semantic_digital_twin.adapters.mujoco_video_recording import (
    MujocoVideoRecorder,
    VideoResolution,
)
from semantic_digital_twin.adapters.multi_sim import MujocoCamera
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.world import World
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from daisy_cable_demo import CableWorkcellLayout

# %% command line of a trial


class Command(StrEnum):
    """
    The commands of the command line of the experiment.
    """

    RECORD = "record"
    """
    Run the demo repeatedly and record the illegal joint states.
    """

    REPLAN = "replan"
    """
    Run the demo repeatedly with replanning around recorded illegal joint states.
    """

    SUMMARIZE = "summarize"
    """
    Summarize a finished experiment, or compare a replanning experiment with the
    experiment of the same seeds without replanning.
    """

    JOINT_STATES = "joint-states"
    """
    Count the illegal joint states of a finished series: logged, duplicated, written,
    and within the guard's reach.
    """

    VIDEO = "video"
    """
    Record one seed with and without replanning and put the videos side by side.
    """

    BROWSE = "browse"
    """
    Open the window listing recorded illegal joint states and keeping chosen ones.
    """

    TRIAL = "trial"
    """
    Run the demo once; used by the other commands to run each trial in a process of its
    own.
    """


class EnvironmentVariable(StrEnum):
    """
    Environment variables the experiment sets for the processes it starts.
    """

    MUJOCO_GRAPHICS_BACKEND = "MUJOCO_GL"
    """
    The OpenGL backend MuJoCo renders with, chosen when MuJoCo is imported.
    """


class TrialOption(StrEnum):
    """
    The options of the :attr:`Command.TRIAL` command.
    """

    NUMBER = "--number"
    SEED = "--seed"
    DIRECTORY = "--directory"
    ILLEGAL_JOINT_STATES = "--illegal-joint-states"
    VIDEO = "--video"


# %% noise


@dataclass(frozen=True)
class TrialNoise:
    """
    How far each trial deviates from the nominal cable demo.
    """

    post_position: float = 0.03
    """
    Largest shift of the cable post in x and y, in metres.
    """

    post_yaw: float = 0.05
    """
    Largest rotation of the cable post about the vertical, in radians.
    """

    def perturbed_layout(
        self, layout: CableWorkcellLayout, random_generator: np.random.Generator
    ) -> CableWorkcellLayout:
        """
        :return: The layout with the post shifted and rotated by uniform noise.
        """
        shift_x, shift_y = random_generator.uniform(
            -self.post_position, self.post_position, size=2
        )
        return replace(
            layout,
            post_x=layout.post_x + shift_x,
            post_y=layout.post_y + shift_y,
            post_yaw=layout.post_yaw
            + random_generator.uniform(-self.post_yaw, self.post_yaw),
        )


# %% one trial


@dataclass
class TrialFiles:
    """
    The files a trial writes, all in one directory.
    """

    directory: Path
    """
    The directory of the files of all trials.
    """

    number: int
    """
    Number of the trial.
    """

    @property
    def collisions(self) -> Path:
        """
        :return: The log of the illegal joint states the trial ran into.
        """
        return self.directory / f"trial_{self.number:03d}.collisions.jsonl"

    @property
    def replanning(self) -> Path:
        """
        :return: The log of the motions the trial replanned.
        """
        return self.directory / f"trial_{self.number:03d}.replanning.jsonl"

    @property
    def output(self) -> Path:
        """
        :return: What the trial's process wrote to its output stream.
        """
        return self.directory / f"trial_{self.number:03d}.log"

    @staticmethod
    def entries_in(log: Path) -> int:
        """
        :return: The number of entries in a log of the trial, none if it never wrote it.
        """
        return len(log.read_text().splitlines()) if log.exists() else 0

    @property
    def errors(self) -> Path:
        """
        :return: What the trial's process wrote to its error stream.
        """
        return self.directory / f"trial_{self.number:03d}.errors.log"


@dataclass
class TrialResult:
    """
    How one trial ended.
    """

    number: int
    """
    Number of the trial.
    """

    seed: int
    """
    The seed the trial's noise was drawn with.
    """

    failure: str | None
    """
    The last line the trial wrote to its error stream if it failed, or ``None`` if it
    completed.
    """

    illegal_joint_states: int
    """
    How many illegal joint states the trial ran into.
    """

    replanned_motions: int
    """
    How many motions the trial replanned.
    """


@dataclass
class CableDemoTrial:
    """
    One run of the cable demo in a newly built, kinematically simulated world whose
    workcell and arm postures are perturbed by seeded noise, logging every self-
    collision of the arms.
    """

    number: int
    """
    Number of the trial.
    """

    seed: int
    """
    The seed the noise is drawn with.
    """

    files: TrialFiles
    """
    The files the trial writes.
    """

    noise: TrialNoise = field(default_factory=TrialNoise)
    """
    How far the trial deviates from the nominal demo.
    """

    illegal_joint_states: list[Path] = field(default_factory=list)
    """
    Databases of illegal joint states the motions are replanned around; none to perform
    the demo as it is.
    """

    video: Path | None = None
    """
    The file a video of the trial is written to, or ``None`` for no video.
    """

    @property
    def command_line(self) -> list[str]:
        """
        :return: The command line that runs this trial in a process of its own.
        """
        arguments = [
            sys.executable,
            "-m",
            "experiments.daisy_cable_replanning",
            Command.TRIAL,
            TrialOption.NUMBER,
            str(self.number),
            TrialOption.SEED,
            str(self.seed),
            TrialOption.DIRECTORY,
            str(self.files.directory),
        ]
        if self.illegal_joint_states:
            arguments += [
                TrialOption.ILLEGAL_JOINT_STATES,
                *(str(path) for path in self.illegal_joint_states),
            ]
        if self.video is not None:
            arguments += [TrialOption.VIDEO, str(self.video)]
        return arguments

    def run_in_own_process(self) -> TrialResult:
        """
        Run the trial in a process of its own, so one trial's world and failures stay
        out of the next.

        :return: How the trial ended.
        """
        self.files.directory.mkdir(parents=True, exist_ok=True)
        with (
            self.files.output.open("w") as output,
            self.files.errors.open("w") as errors,
        ):
            process = subprocess.run(
                self.command_line, stdout=output, stderr=errors, check=False
            )
        # A DataclassException ends its message with a suggestion line; report the error.
        error_lines = [
            line
            for line in self.files.errors.read_text().splitlines()
            if not line.startswith("Suggestion:")
        ]
        result = TrialResult(
            number=self.number,
            seed=self.seed,
            failure=(
                None
                if process.returncode == 0
                else (
                    error_lines[-1]
                    if error_lines
                    else f"exit code {process.returncode}"
                )
            ),
            illegal_joint_states=self.files.entries_in(self.files.collisions),
            replanned_motions=self.files.entries_in(self.files.replanning),
        )
        print(
            f"trial {self.number}: {result.failure or 'completed'}, "
            f"{result.illegal_joint_states} illegal joint states, "
            f"{result.replanned_motions} replanned motions",
            flush=True,
        )
        return result

    def run(self) -> None:
        """
        Build the world and perform every stage of the demo in it.
        """
        random_generator = np.random.default_rng(self.seed)
        # OMPL draws from one generator per process, seeded once before any planner.
        ompl_util.RNG.setSeed(self.seed)
        world = URDFParser.from_file(file_path=DAiSy.get_ros_file_path()).parse()
        DAiSy.from_world(world)
        robot = world.get_semantic_annotations_by_type(DAiSy)[0]
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        motion_tracker = ActiveMotionTracker()
        ArmCollisionRecorder(
            _world=world,
            robot=robot,
            motion_tracker=motion_tracker,
            log=self.files.collisions,
            trial=self.number,
            joint_names=IllegalJointStateDatabase.for_daisy_arms().joint_names,
        )
        demo = DemoScript.daisy_cable_demo().load()
        workcell = demo.CableWorkcell.find_or_add_to(
            world,
            robot,
            self.noise.perturbed_layout(demo.CableWorkcellLayout(), random_generator),
        )
        context = Context(
            world,
            robot,
            plan_transformations=self._plan_transformations(motion_tracker),
        )
        plans = demo.CableDemoPlans(context=context, workcell=workcell)
        recorder = self._start_video(world)
        for stage in demo.CableDemoStage:
            motion_tracker.begin_stage(stage)
            stage_plan = plans.for_stage(stage).plan
            stage_plan.node_callbacks.append(motion_tracker)
            with kinematically_simulated_robot(collision_avoidance=False):
                stage_plan.perform()
        if recorder is not None:
            recorder.stop().write(self.video)

    def _plan_transformations(
        self, motion_tracker: ActiveMotionTracker
    ) -> list[ReplanningTransformation]:
        """
        :return: The transformation replanning around the illegal joint states, if
            there are any to replan around.
        """
        if not self.illegal_joint_states:
            return []
        return [
            ReplanningTransformation(
                illegal_joint_states=IllegalJointStateDatabase.load_merged(
                    self.illegal_joint_states
                ),
                parameters=ReplanningParameters(),
                replanning_log=ReplanningLog(
                    path=self.files.replanning,
                    trial=self.number,
                    motion_tracker=motion_tracker,
                ),
            )
        ]

    def _start_video(self, world: World) -> MujocoVideoRecorder | None:
        """
        :return: A recorder capturing the world from now on, if a video is wanted.
        """
        if self.video is None:
            return None
        resolution = VideoResolution(width=640, height=480)
        recorder = MujocoVideoRecorder(
            world=world,
            capture_every_n_state_changes=8,
            resolution=resolution,
            camera=self._work_area_camera(world, resolution),
        )
        recorder.start()
        return recorder

    @staticmethod
    def _work_area_camera(world: World, resolution: VideoResolution) -> MujocoCamera:
        """
        :return: A camera, newly attached to the world's root, looking diagonally down
            at the arms and the cable hanger.
        """
        work_area = np.array([[-0.1, -0.2, 0.6], [0.9, 1.3, 1.5]])
        pose = MujocoCamera.overview_pose(work_area, distance_factor=0.75)
        # MuJoCo orders the quaternion scalar first, Quaternion.to_np scalar last.
        x, y, z, w = pose.quaternion.to_np().tolist()
        camera = MujocoCamera(
            name="cable_demo_camera",
            body=world.root,
            position=pose.position.to_np()[:3].tolist(),
            quaternion=[w, x, y, z],
            resolution=[float(resolution.width), float(resolution.height)],
        )
        world.root.add_simulator_property(camera)
        return camera


# %% experiment


@dataclass
class CableDemoExperiment:
    """
    Runs the cable demo a number of times, each trial in a process of its own, and
    collects the illegal joint states of all trials into one database.
    """

    output_directory: Path
    """
    Directory the trial files, the trial results and the database are written to.
    """

    trials: int = 100
    """
    How many times the demo is run.
    """

    first_seed: int = 0
    """
    Seed of the first trial; each further trial uses the next one.
    """

    workers: int = 2
    """
    How many trials run at the same time.
    """

    illegal_joint_states: list[Path] = field(default_factory=list)
    """
    Databases of illegal joint states the motions are replanned around; none to perform
    the demo as it is.
    """

    removes_duplicate_poses: bool = True
    """
    Whether states that repeat an earlier pose are left out of the database written
    after all trials; the trials' own logs keep every state.
    """

    duplicate_tolerance: float = 0.02
    """
    Largest joint difference, in radians, at which two states with the same colliding
    joints count as the same pose, the spacing the recorder logs states at.
    """

    @property
    def database_path(self) -> Path:
        """
        :return: The file the illegal joint states of all trials are written to.
        """
        return self.output_directory / "illegal_joint_states.json"

    @property
    def results_path(self) -> Path:
        """
        :return: The file the result of every trial is written to.
        """
        return self.output_directory / "trial_results.json"

    def run(self) -> list[TrialResult]:
        """
        Run every trial, then write the trial results and the database of the illegal
        joint states they ran into.

        :return: The result of every trial.
        """
        trial_directory = self.output_directory / "trials"
        trial_directory.mkdir(parents=True, exist_ok=True)
        trials = [
            CableDemoTrial(
                number=number,
                seed=self.first_seed + number,
                files=TrialFiles(directory=trial_directory, number=number),
                illegal_joint_states=self.illegal_joint_states,
            )
            for number in range(self.trials)
        ]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(CableDemoTrial.run_in_own_process, trials))

        self.results_path.write_text(json.dumps(to_json(results)))
        self.collected_database(trials).save(self.database_path)
        return results

    def collected_database(
        self, trials: list[CableDemoTrial]
    ) -> IllegalJointStateDatabase:
        """
        :return: The illegal joint states the trials logged, without repeated poses if
            :attr:`removes_duplicate_poses`.
        """
        database = IllegalJointStateDatabase.for_daisy_arms()
        for trial in trials:
            if trial.files.collisions.exists():
                database.extend_from_log(trial.files.collisions)
        if not self.removes_duplicate_poses:
            return database
        distinct = database.without_duplicate_poses(self.duplicate_tolerance)
        print(
            f"{len(database.illegal_joint_states) - len(distinct.illegal_joint_states)}"
            f" of {len(database.illegal_joint_states)} illegal joint states repeat a "
            f"pose within {self.duplicate_tolerance} rad and were left out",
            flush=True,
        )
        return distinct
