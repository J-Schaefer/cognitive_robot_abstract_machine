"""
The DAiSy cable demo, performed once in its nominal workcell, with Giskard avoiding
collisions and every Cartesian motion that gets stuck replanned in this process, as in
series 4 of the experiment, but without trial noise.

The motion charts hold no planner, so they run on the real robot as well as in the
kinematic simulation of this process::

    python -m experiments.daisy_cable_replanning.stuck_motion_demo --execution simulated

Semi-real and real runs need a running DAiSy controller::

    ros2 launch giskardpy_ros giskardpy_daisy_standalone.launch.py
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from coraplex.execution_environment import ExecutionEnvironment
from experiments.daisy_cable_replanning.cable_demo import DemoScript
from experiments.daisy_cable_replanning.collisions import ActiveMotionTracker
from experiments.daisy_cable_replanning.replanning import (
    ReplanningLog,
    ReplanningParameters,
)
from experiments.daisy_cable_replanning.stuck_motion_replanning import (
    StuckMotionReplanning,
)
from experiments.daisy_gripper_demo.demo import ExecutionChoice
from giskardpy.middleware.ros2 import rospy

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

    def run(self) -> None:
        """
        Set up the robot and the workcell and perform every stage of the demo, avoiding
        collisions in every stage but homing.
        """
        demo = DemoScript.daisy_cable_demo().load()
        _, world, robot, context = (
            demo.setup_sim_daisy()
            if self.execution is ExecutionChoice.SIMULATED
            else demo.setup_real_daisy()
        )
        with ExitStack() as ros_runtime:
            # Without it, the ROS spin thread aborts the interpreter at exit.
            ros_runtime.callback(rospy.shutdown)
            motion_tracker = ActiveMotionTracker()
            context.plan_transformations.append(
                StuckMotionReplanning(
                    parameters=self.parameters,
                    replanning_log=ReplanningLog(
                        path=self.replanning_log, trial=0, motion_tracker=motion_tracker
                    ),
                )
            )
            workcell = demo.CableWorkcell.find_or_add_to(
                world, robot, demo.CableWorkcellLayout()
            )
            plans = demo.CableDemoPlans(context=context, workcell=workcell)
            for stage in demo.CableDemoStage:
                print(f"Performing the {stage} stage.", flush=True)
                motion_tracker.begin_stage(stage)
                stage_plan = plans.for_stage(stage).plan
                stage_plan.node_callbacks.append(motion_tracker)
                with ExecutionEnvironment(
                    self.execution.execution_type,
                    collision_avoidance=stage is not demo.CableDemoStage.HOME,
                ):
                    stage_plan.perform()
        print("Cable demo finished.", flush=True)


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
        default=ExecutionChoice.SIMULATED,
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
