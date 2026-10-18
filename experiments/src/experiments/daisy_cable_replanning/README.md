# DAiSy cable demo: recording illegal joint states and replanning around them

This experiment runs the DAiSy cable demo (`coraplex/demos/daisy_cable_demo/demo.py`)
many times in kinematic simulation, without collision avoidance. It has two parts:

1. **Recording.** It records every joint state in which DAiSy collides with itself:
   the arms with each other, an arm with itself, or an arm with the rest of the
   robot. These are the *illegal joint states*.
2. **Replanning.** Giskard performs every Cartesian motion as before, while a monitor
   watches how close the arms get to an illegal joint state. Once they are close, the
   motion is stopped and the arm retracts to the last safe joint state. It then
   reaches the motion's target along a joint path that OMPL's RRT-Connect plans
   around the illegal joint states and collisions. The planned path is handed back to
   Giskard for execution.

Recording and replanning alternate: illegal joint states that a replanned run still
runs into are added to the database for the next round.

## Contents

| file | what it holds |
|---|---|
| `cable_demo.py` | `DemoScript`, which imports `demo.py` as a module, so its workcell and plans are used as they are |
| `trials.py` | `CableDemoTrial`, one run of the demo in its own process; `CableDemoExperiment`, many runs; `TrialNoise` |
| `collisions.py` | recording: `ArmCollisionRecorder`, `IllegalJointState`, `IllegalJointStateDatabase`, `MotionLabel` |
| `replanning.py` | replanning: `ReplanningTransformation`, `ReplannedToolCenterPointMotion`, `JointSpacePlanner`, `GoalConfigurationSampler`, `ReplanningParameters` |
| `summary.py` | `ExperimentOutcome` and `ReplanningComparison`, the numbers reported below |
| `curation.py` | `IllegalJointStateCuration`, choosing recorded states to keep; `KeptDatabase`, the permanent files |
| `joint_state_browser.py`, `robot_snapshot.py` | the window for choosing states to keep, and the rendering of the robot in it |
| `command_line.py`, `__main__.py` | the command line: `record`, `replan`, `summarize`, `video`, `browse`, `trial` |
| `run_series.sh` | one complete series, as run for the results below |

The generic Giskard pieces live in giskardpy. They are documented in
`giskardpy/doc/examples/replanning.md`:
- `CloseToJointConfigurations`, a monitor
- `ReplacedWhenTrue`, a goal
- `JointWaypointPath` and `RetractToLastSafeJointConfiguration`, tasks

## How a replanned motion runs

`ReplanningTransformation` is a coraplex plan transformation. It replaces each
`MoveToolCenterPointMotion` of the actions that illegal joint states were recorded in
(`CableGraspAction`, `CableRegraspAction`, `CableRehangAction`) with a
`ReplannedToolCenterPointMotion`. The motion chart of that motion is:

```
ReplacedWhenTrue
├── monitor:     CloseToJointConfigurations
│                observes True once the arms are within trigger_distance of an
│                illegal joint state, and remembers the last joint state that was
│                at least safe_distance from all of them
├── monitored:   the original MoveTCP task, i.e. Giskard's optimization-based motion
└── replacement: Sequence
                 ├── RetractToLastSafeJointConfiguration  (back to the remembered state)
                 └── PlannedJointPathToPose
                     1. GoalConfigurationSampler: inverse kinematics from the current
                        joint state and goal_seeds random seeds gives the distinct
                        joint configurations that reach the target pose
                     2. JointSpacePlanner: OMPL RRT-Connect in the six joints of the
                        moving arm (the other arm is held still), to the valid goal
                        configuration nearest to the current joint state first, then
                        to the next nearest. A joint state is valid if
                          - the arm's bodies keep collision_margin from the robot and
                            the workcell (the soft cable is ignored), and
                          - it keeps clearance from every illegal joint state
                     3. the simplified path, as waypoints at most waypoint_spacing
                        apart, is followed by Giskard
```

The distance to an illegal joint state is the largest joint difference, counting only
the joints of the arm or arms whose bodies collided in that state.

## Prerequisites

- ROS 2 Jazzy and the workspace with `iai_daisy_description` and `griplink_interfaces`
  sourced:
  ```bash
  source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
  ```
- The repository's Python environment with the `experiments` package installed
  (`pip install -e experiments`). This also installs `ompl` (2.0.1 or newer, Python
  wheels on PyPI).
- `systemd-run --user` to cap memory. A run uses about 0.6 to 1.5 GB per worker.

> [!WARNING]
> Never cap memory with `ulimit -v`. The video's OpenGL context needs more address
> space than such a cap leaves it, and MuJoCo then fails with `Shadow framebuffer is
> not complete`.

> [!WARNING]
> Don't edit the package's Python files while an experiment runs. Every trial is a
> new process that imports them again.

## Repeating the experiment

### One command per series

```bash
cd <repository root>
experiments/src/experiments/daisy_cable_replanning/run_series.sh \
    experiments/results/daisy_cable_replanning/<series name> 3 \
    [databases of earlier series ...]
```

`TRIALS`, `WORKERS`, `MEMORY` and `HELDOUT_SEED` can be set as environment variables.
The script writes `summary.txt` and `video/` into the series directory. With 6 workers
on a 16-core machine, a series of 100 trials per step takes about 1.5 hours.

### Step by step

```bash
E="python -m experiments.daisy_cable_replanning"
CAP="systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0"
S=experiments/results/daisy_cable_replanning/my_series

# 1. record on the training seeds, and the baseline on the held-out seeds
$CAP $E record --output $S/record         --first-seed 0   --trials 100 --workers 6
$CAP $E record --output $S/heldout_record --first-seed 100 --trials 100 --workers 6

# 2. replanning rounds on the training seeds; each round uses every database so far
$CAP $E replan --output $S/replan_round_1 --first-seed 0 --trials 100 --workers 6 \
    --illegal-joint-states $S/record/illegal_joint_states.json
$CAP $E replan --output $S/replan_round_2 --first-seed 0 --trials 100 --workers 6 \
    --illegal-joint-states $S/record/illegal_joint_states.json \
                           $S/replan_round_1/illegal_joint_states.json
# ... until a round runs into few new illegal joint states

# 3. evaluate on the held-out seeds with all databases
$CAP $E replan --output $S/heldout_replan --first-seed 100 --trials 100 --workers 6 \
    --illegal-joint-states $S/record/illegal_joint_states.json \
                           $S/replan_round_*/illegal_joint_states.json

# 4. numbers and video
$E summarize --output $S/heldout_replan --baseline $S/heldout_record
$CAP $E video --output $S/video --baseline $S/heldout_record \
    --replanning $S/heldout_replan --illegal-joint-states <same databases as step 3>

# a single trial, e.g. to debug a seed
$E trial --number 0 --seed 117 --directory /tmp/trial \
    [--illegal-joint-states ...] [--video /tmp/trial/run.mp4]
```

The seed determines a trial. It seeds both the workcell noise
(`numpy.random.default_rng(seed)`) and OMPL (`ompl.util.RNG.setSeed(seed)`, once per
trial process), and path simplification does a fixed amount of work rather than running
for a time.

- A trial **without replanning** reproduces exactly: the same seed records the same
  joint states, value for value.
- A trial **with replanning** reproduces only statistically. The first planned path
  matches to about 1e-15 rad, but floating-point differences of that size are enough
  to change a later RRT-Connect search. A seed can then be replanned along a different
  path and end differently. Compare runs of the same seeds by their counts, not seed
  by seed.

## What varies between trials (`TrialNoise`)

The kinematic simulation is deterministic, so every trial draws seeded noise. The cable
post is shifted by up to `post_position` = 3 cm in x and y, and rotated by up to
`post_yaw` = 0.05 rad. Everything that depends on the hanger's pose moves with it.
Every trial starts with both arms parked.

## Parameters (`ReplanningParameters`)

| parameter | value | meaning |
|---|---|---|
| `trigger_distance` | 0.15 rad | closer than this to an illegal joint state stops the Giskard motion |
| `safe_distance` | 0.3 rad | a joint state at least this far from all of them is safe to retract to |
| `clearance` | 0.2 rad | the planned path and its goal keep this far from every illegal joint state (above the trigger, so the next motion does not start triggered) |
| `collision_margin` | 2 mm | the planned path keeps the arm's bodies this far from the robot and the workcell |
| `validity_checking_resolution` | 0.2 % | segment check spacing, as a fraction of the joint space's extent |
| `planning_time` | 10 s | RRT-Connect time limit per replanning |
| `waypoint_spacing` | 0.1 rad | largest joint step between waypoints handed to Giskard |
| `goal_seeds` | 64 | random inverse-kinematics seeds besides the current joint state |
| `goal_seed_spread` | π | largest joint offset of a seed |
| `goal_sampling_seed` | 0 | seed of the seeds |
| `distinct_goal_distance` | 0.05 rad | goal configurations closer than this count as one |

## Output files

Every `record` or `replan` run writes into its output directory:

| file | content |
|---|---|
| `trial_results.json` | per trial: seed, failure (the error's last line, or none), number of illegal joint states, number of replanned motions |
| `illegal_joint_states.json` | the `IllegalJointStateDatabase` of all illegal joint states the trials ran into. Pass it to `--illegal-joint-states` |
| `trials/trial_NNN.collisions.jsonl` | one `IllegalJointState` per line: the 12 arm joint positions, the joints of the colliding arms, the collision kind, the two bodies, the penetration depth, the motion (`MotionLabel`: stage, index in the stage, action, motion type) |
| `trials/trial_NNN.replanning.jsonl` | one replanned motion per line: its label, where the arm retracted to, the planned waypoints |
| `trials/trial_NNN.log`, `.errors.log` | the trial process's output and error streams |

All JSON is written by `krrood.adapters.json_serializer`. `IllegalJointStateDatabase.load`
reads a database back.

## Keeping illegal joint states for good

The databases in `experiments/results/` belong to single runs. Illegal joint states
worth keeping for this robot setup go into a permanent file, by default
`resources/daisy_cable_workcell_illegal_joint_states.json` next to this README
(`KeptDatabase.DAISY_CABLE_WORKCELL`). It has the same format as every recorded
database, so it can be passed to `--illegal-joint-states` like any other.

To choose the states, open the browser:

```bash
python -m experiments.daisy_cable_replanning browse \
    [--recorded experiments/results/.../record/illegal_joint_states.json] \
    [--permanent <permanent file, default: the one above>]
```

1. Enter or browse for the **recorded file** and the **permanent file**, then press
   **Load**. A permanent file that does not exist yet counts as empty. Recorded states
   the permanent file already holds are checked from the start, and marked
   *In permanent file*.
2. Click a row to see its joint state with the robot.
   - Drag to orbit the camera, scroll to zoom.
   - **Focus on contact** looks closely at the deepest contact.
   - **Whole scene** looks at everything again.
   - Below the image: the motion and trial the state was recorded in, the colliding
     bodies, and which bodies touch in this state next to the *nominal* workcell (the
     trial's workcell was shifted by noise).
3. Check **Keep** for the states to keep, or highlight rows (shift- or ctrl-click,
   sortable by every column) and press **Keep highlighted** or **Drop highlighted**.
   **Keep all** and **Keep none** apply to every row.
4. Press **Save to permanent file**. The permanent file then holds:
   - every state it held that the recorded file does not contain, and
   - every checked recorded state.

   Unchecking a recorded state that was in the permanent file removes it. Both files
   are loaded again, so the table shows the new content.

Two records are the same joint state when their positions agree to a nanoradian.

The command starts itself again with `MUJOCO_GL=egl` if no MuJoCo backend is set,
because only EGL renders offscreen next to the Qt window.

## Results

### Series 1: demo with safe intermediate postures

Directory: `series_1_with_safe_poses/`.

In this series, homing moved the left arm to a safe posture, and regrasping and
rehanging first moved the right arm to one. Each trial added up to 0.1 rad of noise to
every joint of those postures.

- Recording, seeds 0-99:
  - 63 trials collided, recording 1,449 illegal joint states.
  - All of them were the same collision: the right forearm against its own gripper's
    side cylinder, in the first Cartesian motion of `CableRehangAction`.
- Replanning: every colliding trial was replanned, became collision-free and
  completed, both on the training seeds (63 of 63) and on held-out seeds 100-149
  (30 of 30). No trial was replanned without need.

### Series 2: demo without safe intermediate postures

Directory: `series_2_without_safe_poses/`.

The safe postures were removed from the demo (commit `704b17119`): homing parks both
arms, and regrasping and rehanging start wherever the arms are. The replanning used
series 1's database in addition.

| run | seeds | completed | trials colliding | completed without any collision | illegal joint states | replanned motions |
|---|---|---|---|---|---|---|
| `record` | 0-99 | 79 | 94 | 0 | 8,930 | - |
| `replan_round_1` ¹ | 0-99 | 70 | 83 | 0 | 11,756 | 177 |
| `replan_round_2` | 0-99 | 58 | 21 | 40 | 1,141 | 468 |
| `replan_round_3` ² | 0-99 | 20 | 12 | 17 | 104 | 456 |
| `heldout_record` | 100-199 | 76 | 87 | 0 | 8,260 | - |
| `heldout_replan` ³ | 100-199 | 61 | 60 | 3 | 4,607 | 332 |

¹ Run with an earlier planner. That planner built its collision matrix when the
motion chart was built, before the cable was reattached, and treated every contact the
collision detector reported as a collision. Its illegal joint states are valid
recordings; its replanning numbers are not comparable with the later rows.
² Run with 16 goal seeds; `heldout_replan` uses 64.
³ The current code. Two earlier variants of this evaluation are kept for comparison:

| variant (directory) | completed | trials colliding | completed without any collision | illegal joint states |
|---|---|---|---|---|
| any reachable goal, OMPL unseeded (`heldout_replan_any_goal`) | 54 | 58 | 17 | 8,940 |
| nearest goal first, OMPL unseeded (`heldout_replan_unseeded`) | 60 | 61 | 3 | 4,903 |
| nearest goal first, OMPL seeded (`heldout_replan`, current) | 61 | 60 | 3 | 4,607 |

With any reachable goal, the planned path often ends in a joint configuration far from
Giskard's own. The following motions then run through regions no training round saw,
including 579 contacts between the right upper arm and the table. Preferring the
nearest goal halves the illegal joint states. Most of the remaining ones are in the
first two motions of the rehang (`rehang#0`, `rehang#1`, 57 trials), which few
training trials reached in the later rounds, so fewer trials complete without any
collision. One run per variant is too few to rank the variants on that number.

What the series shows:

- **Collisions.** Without the safe postures the demo collides in 94 of 100 trials,
  mostly the right forearm against its own gripper during regrasping and rehanging.
  168 collisions are between the arms (right forearm against the left gripper or
  wrist), during the grasp. Each replanning round removes most of the remaining
  collisions: 8,930 → 1,141 → 104 illegal joint states on the training seeds.
- **New collisions after replanning.** A replanned motion leaves the arm in a
  different joint configuration than Giskard would have. The following Cartesian
  motions then run through joint states no earlier round has seen. The guard only knows
  recorded states, so it cannot stop those. This is why the rounds are needed, and why
  some collisions remain.
- **Failures.** Each round's database makes more goal configurations invalid. 64
  inverse-kinematics seeds instead of 16 find valid goals in more cases, but runs still
  end with `NoJointPathFound` when no valid goal configuration exists for a target pose.
  Independently of replanning, 15 to 21 trials per 100 stall in a Cartesian motion
  during regrasping (`MotionMadeNoProgress`); the baseline stalls there too.
- **Generalization.** On the held-out seeds, replanning removes about half of the
  illegal joint states (8,260 → 4,607) and cuts the colliding trials from 87 to 60. It
  removes far fewer than on the training seeds, because the database covers the
  configurations the training runs visited, not those of new seeds.
- **Planning.** RRT-Connect solved the problems that had a valid goal in 0.03 to
  2.5 s.

`series_2_without_safe_poses/video/daisy_cable_replanning_seed106.mp4`, also in the
repository's `video/` directory, shows held-out seed 106. It was chosen by the `video`
command as the first held-out seed that collided without replanning and completed
without any collision with it:

- Left, Giskard alone: 59 illegal joint states.
- Right, with replanning: 6 motions replanned, no collision, every stage completed.

## Known limitations

- The guard is joint-space proximity to *recorded* states. It does not foresee
  collisions in regions no trial has visited; a distance-based monitor on the collision
  checker would.
- `plan_waypoints` (inverse kinematics plus planning) runs inside one control cycle.
  That is fine in simulation, but on the real robot it would stall the controller for
  up to several seconds.
- The cable is ignored by the planner. It is soft and meant to touch the hanger and
  the grippers.
- In the videos, the cable stays on the post while the grippers carry it. The MuJoCo
  mirror does not follow reattachments; the simulated world does.
- The JSON files of series 1 originally stored the stage of a `MotionLabel` as an
  enum of a module that no longer exists. They were migrated to the plain stage name.
