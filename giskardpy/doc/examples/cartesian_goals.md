---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Cartesian Goals Example

This example demonstrates how to use `CartesianPose` goals to move a robot to a specific pose.

```{code-cell} ipython3
from giskardpy.executor import SimulationPacer, Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3, Quaternion

# 1. Create a World and a Robot (using dummy world for this example)
world = World()
# In a real scenario, you would load a robot into the world here.

# 2. Create a Motion Statechart
msc = MotionStatechart()

# 3. Define a Cartesian Pose Goal
# Note: root_link and tip_link should be KinematicStructureEntity objects from the world.
# For demonstration, we assume they exist in your world.
# goal_pose = Pose(position=Vector3(1, 0, 0), orientation=Quaternion(0, 0, 0, 1), reference_frame=world.root)
# goal = CartesianPose(name="move_to_goal", root_link=world.root, tip_link=robot_tip, goal_pose=goal_pose)

# msc.add_node(goal)
# msc.add_node(EndMotion.when_true(goal))

# 4. Set up the Executor
kin_sim = Executor(
    context=MotionStatechartContext(
        world=world,
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    ),
    pacer=SimulationPacer(real_time_factor=1.0),
)

# kin_sim.compile(msc)
# kin_sim.tick_until_end(timeout=1000)
```
