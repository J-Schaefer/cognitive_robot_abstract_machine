# DAiSy cable demo: recording illegal joint states and replanning around them

Runs the cable demo of `coraplex/demos/daisy_cable_demo` (home, grasp the cable from the
hanger, regrasp it above the table, hang it back) many times in kinematic simulation.
The workcell and the plans come from the demo script itself: `DemoScript` imports
`demo.py` as a module, without running its `__main__` block, and uses its
`CableWorkcell`, `CableWorkcellLayout`, `SafeArmPosture`, `CableDemoStage` and
`CableDemoPlans`.
It first records the joint states in which DAiSy collides with itself, and then
replans the motions that approach those joint states.

## Running

ROS 2 and the workspace with `iai_daisy_description` and `griplink_interfaces` have to
be sourced:

```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash

# 1. record: 100 trials, collisions without any avoidance
python -m experiments.daisy_cable_replanning record \
    --output results/record --trials 100 --workers 2

# 2. replan: the same seeds, motions replanned around the recorded states
python -m experiments.daisy_cable_replanning replan \
    --output results/replan --trials 100 --workers 2 \
    --illegal-joint-states results/record/illegal_joint_states.json

# one trial, with a video
python -m experiments.daisy_cable_replanning trial --number 0 --seed 5 \
    --directory results/video \
    --illegal-joint-states results/record/illegal_joint_states.json \
    --video results/video/replanning.mp4
```

Every trial runs in a process of its own. Each trial writes these files to `trials/`:
- `trial_NNN.collisions.jsonl`: one illegal joint state per line
- `trial_NNN.replanning.jsonl`: one replanned motion per line
- its output and error logs

The experiment writes `trial_results.json` and `illegal_joint_states.json`.

Cap memory with a cgroup (`systemd-run --user --scope -p MemoryMax=4G ...`), not with
`ulimit -v`: the OpenGL driver used for the video reserves more address space than a
`ulimit -v` cap leaves it, and fails with `Shadow framebuffer is not complete`.

## What varies between trials

The kinematic simulation is deterministic, so each trial draws seeded noise
(`TrialNoise`):
- the cable post is shifted by up to 3 cm and rotated by up to 0.05 rad
- each joint of the two safe arm postures is offset by up to 0.1 rad

Each trial starts with both arms parked.

## Recording

`ArmCollisionRecorder` checks, after every change of the world state, every pair of
robot bodies involving an arm body that DAiSy's collision rules check
(`ArmCollisionPairs`). It does so with Bullet, for contact only. A contact is logged as
an `IllegalJointState`, which holds:
- the positions of all twelve arm joints
- the joints of the arms whose bodies touch, since only those decide closeness to the
  state
- the collision kind (between the arms, within one arm, arm and robot body)
- the bodies
- the penetration depth
- the motion that was running (`MotionLabel`)

Objects attached to the robot later, such as the cable or the hanger, are not robot
bodies.

## Replanning

`ReplanningTransformation` is a plan transformation. It replaces the
`MoveToolCenterPointMotion`s of every action that illegal joint states were recorded in
with `ReplannedToolCenterPointMotion`. Its motion chart is one Giskard chart:

```
ReplacedWhenTrue
├── monitor:      CloseToJointConfigurations  (closer than 0.15 rad to an illegal state)
├── monitored:    the original MoveTCP task (Giskard's optimization-based motion)
└── replacement:  Sequence
                  ├── RetractToLastSafeJointConfiguration  (last state >= 0.3 rad away)
                  └── PlannedJointPathToPose               (IK goal + OMPL RRT-Connect)
```

When the monitor fires, the Cartesian motion ends and the arm retracts to the last joint
state that was at least 0.3 rad from every illegal state. From there:
1. `PlannedJointPathToPose` solves IK for the original target.
2. It plans a joint path with OMPL that is collision-free (margin 5 mm against the robot
   and the workcell) and keeps 0.1 rad from every illegal state.
3. It hands the path back to Giskard as waypoints that a joint task follows.

The giskardpy nodes are generic and documented in
`giskardpy/doc/examples/replanning.md`.

## Choice of motion planner

Giskard optimizes locally, one control cycle at a time, so it cannot find a way around
a collision it is already heading into. The alternative motion therefore needs a
global, sampling-based planner. The options considered:

| Planner | Verdict |
|---|---|
| **OMPL**, RRT-Connect (`pip install ompl`, nanobind bindings since 2.0) | **used**: Python wheels, mature RRT-Connect, path simplification, takes any state validity callback, so the world's own Bullet collision checking and the illegal-state distance plug in directly |
| MoveIt 2 | uses OMPL itself, but needs its own robot model and planning scene and a running ROS stack; duplicates the world model |
| VAMP (SIMD-accelerated) | very fast, but needs a robot-specific compiled collision model; DAiSy has none |
| cuRobo | needs CUDA; this machine has no NVIDIA GPU |
| own RRT | no dependency, but no path simplification and less robust |

The planner works on the six joints of the moving arm, with the other arm held where it
is. A path is checked at a resolution of 0.5 % of the extent of the joint space, simplified,
and interpolated to waypoints at most 0.1 rad apart.

## Results

Recording run: seeds 0-99, kinematic simulation, no collision avoidance.

- 63 of 100 trials collided; 1449 illegal joint states were recorded.
- Every one of them was the same collision within the right arm: the forearm reaching
  into the gripper's side cylinder during the first tool center point motion of
  `CableRehangAction` (`rehang#2`).
- No collision between the two arms, or between an arm and the rest of the robot,
  occurred.
- 6 trials failed independently of collisions, because a Cartesian motion stalled at a
  pose the noise put out of reach.

Replanning with that database:

| seeds | trials colliding without replanning | of those replanned / collision-free / completed | replanned without need | collisions with replanning |
|---|---|---|---|---|
| 0-99 (seen) | 63 | 63 / 63 / 63 | 0 of 31 | 0 |
| 100-149 (held out) | 30 | 30 / 30 / 30 | 0 of 14 | 0 |

Both runs fail on the same seeds, from the same stalls as the recording. RRT-Connect
found every path, in 29 ms median and 0.2 s at most.

`video/daisy_cable_replanning_seed103.mp4` shows held-out seed 103 side by side: Giskard
alone on the left, with replanning on the right.

.. note:: In the video, the cable stays on the post while the grippers carry it. The
   MuJoCo mirror does not follow the cable being reattached to a gripper. The simulated
   world does.
