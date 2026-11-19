"""
The DAiSy cable demo, performed once in its nominal workcell, with Giskard avoiding
collisions and every Cartesian motion that gets stuck replanned in this process, as in
series 4 of the experiment, but without trial noise.

The statecharts hold no planner, so they run on a controller elsewhere as well as in the
kinematic simulation of this process::

    python -m experiments.daisy_cable_replanning.stuck_motion_demo --execution simulated

Runs against a controller need a running DAiSy controller::

    ros2 launch giskardpy_ros giskardpy_daisy_standalone.launch.py
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.plan_transformation import PlanRewriting
from cramph.composites import ChildChooserAccess
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.daisy_cable_replanning.cable_demo import DemoScript
from experiments.daisy_cable_replanning.collisions import ActiveMotionTracker
from experiments.daisy_cable_replanning.replanning import (
    ReplanningLog,
    ReplanningParameters,
)
from experiments.daisy_cable_replanning.stuck_motion_replanning import (
    JointPathPlanning,
    StuckMotionReplanning,
)
from experiments.daisy_gripper_demo.demo import ExecutionChoice
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelReloadSynchronizer,
    WorldSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.world import World

# %% demo


@dataclass
class StuckMotionCableDemo:
    """
    Homes DAiSy, grasps the cable, regrasps it and hangs it back, replanning the
    Cartesian motions that collision avoidance leaves stuck.
    """

    execution: ExecutionChoice
    """
    Where the plans are performed.
    """

    replanning_log: Path
    """
    The file every planned path is appended to.
    """

    parameters: ReplanningParameters = field(default_factory=ReplanningParameters)
    """
    When a motion counts as stuck and how the path replacing it is planned.
    """

    ros_node_name: str = "daisy_cable_replanning_node"
    """
    The name of the ROS node a run against a controller communicates through.
    """

    def run(self) -> None:
        """
        Set up the robot and the workcell and perform every stage of the demo, avoiding
        collisions in every stage but homing.
        """
        with ExitStack() as ros_runtime:
            ros_session = self._start_ros_session()
            if ros_session is not None:
                ros_runtime.callback(ros_session.stop)
            world = self._world(ros_session)
            robot = world.get_semantic_annotations_by_type(DAiSy)[0]
            demo = DemoScript.daisy_cable_demo().load()
            workcell = demo.CableWorkcell.find_or_add_to(
                world, robot, demo.CableWorkcellLayout()
            )
            plans = demo.CableDemoPlans(robot=robot, workcell=workcell)
            motion_tracker = ActiveMotionTracker()
            replanning_log = ReplanningLog(
                path=self.replanning_log, trial=0, motion_tracker=motion_tracker
            )
            for stage in demo.CableDemoStage:
                print(f"Performing the {stage} stage.", flush=True)
                motion_tracker.begin_stage(stage)
                executor = self.execution.executor_type(
                    context=StatechartContext.with_extensions(
                        world=world,
                        extensions=[
                            RobotAccess(robot=robot),
                            PlanRewriting(
                                transformations=[
                                    StuckMotionReplanning(
                                        parameters=self.parameters,
                                        replanning_log=replanning_log,
                                    )
                                ]
                            ),
                        ],
                    ),
                    ros_node=None if ros_session is None else ros_session.node,
                    collision_avoidance=stage is not demo.CableDemoStage.HOME,
                    extensions=[motion_tracker],
                )
                executor.context.require_extension(
                    extension_type=ChildChooserAccess
                ).choosers.append(
                    JointPathPlanning(
                        parameters=self.parameters, replanning_log=replanning_log
                    )
                )
                statechart = Statechart(context=executor.context)
                statechart.add_node(node=plans.for_stage(stage))
                try:
                    executor.compile(statechart=statechart)
                    executor.execute()
                finally:
                    executor.close()
        print("Cable demo finished.", flush=True)

    def _start_ros_session(self) -> RobotDemonstrationRosSession | None:
        """
        :return: The ROS session a run against a controller communicates through, None
            for a simulated run.
        """
        if self.execution is ExecutionChoice.SIMULATED:
            return None
        return RobotDemonstrationRosSession.start(self.ros_node_name)

    @staticmethod
    def _world(ros_session: RobotDemonstrationRosSession | None) -> World:
        """
        :param ros_session: The session of a run against a controller, None for a
            simulated run.
        :return: The world holding DAiSy: parsed from its description for a simulated
            run, else fetched from the controller and kept in sync with it.
        """
        if ros_session is None:
            world = URDFParser.from_file(
                file_path=DAiSy.get_ros_file_path()
            ).create_world()
            DAiSy.from_world(world)
            return world
        world = ros_session.fetch_world()
        WorldSynchronizer(_world=world, node=ros_session.node)
        ModelReloadSynchronizer(_world=world, node=ros_session.node)
        return world


# %% command line


def run_from_command_line() -> None:
    """
    Parse the command line and run the demo.
    """
    parser = argparse.ArgumentParser(
        description="DAiSy cable demo, replanning motions that get stuck."
    )
    parser.add_argument(
        "--execution",
        type=ExecutionChoice,
        choices=list(ExecutionChoice),
        default=ExecutionChoice.STANDALONE,
        help="where the plans are performed",
    )
    parser.add_argument(
        "--replanning-log",
        type=Path,
        default=Path("stuck_motion_replanning.jsonl"),
        help="file every planned path is appended to",
    )
    arguments = parser.parse_args()
    StuckMotionCableDemo(
        execution=arguments.execution, replanning_log=arguments.replanning_log
    ).run()


if __name__ == "__main__":
    run_from_command_line()
