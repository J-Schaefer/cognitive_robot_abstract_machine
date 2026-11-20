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
| `collisions.py` | recording: `ArmCollisionRecorder`, `IllegalJointState`, `IllegalJointStateDatabase`, `MotionLabel`, `ActiveMotionTracker` (an executor extension naming the running motion) |
| `replanning.py` | replanning: `ReplanningTransformation`, `IllegalJointStateReplanning`, `StagnationReplanning`, `RecoveryBeforeStuckMotion`, `PlannedJointPathToPose`, `PathSinceMotionStart`, `PathSinceLastClearJointState`, `RetractAlongRememberedPath`, `JointSpacePlanner`, `GoalConfigurationSampler`, `ReplanningParameters` |
| `inverse_kinematics.py` | `InverseKinematicsSolver`, the QP-based inverse kinematics `GoalConfigurationSampler` uses (moved here from semantic_digital_twin, which no longer has it; needs `daqp`) |
| `summary.py` | `ExperimentOutcome` and `ReplanningComparison`, the numbers reported below |
| `joint_state_metrics.py` | `SeriesJointStateMetrics`, the joint state counts of a series (`joint-states` command) |
| `curation.py` | `IllegalJointStateCuration`, choosing recorded states to keep; `KeptDatabase`, the permanent files |
| `joint_state_browser.py`, `robot_snapshot.py` | the window for choosing states to keep, and the rendering of the robot in it |
| `stuck_motion_replanning.py`, `stuck_motion_demo.py` | `StuckMotionReplanning`, `JointPathToPose` and `JointPathPlanning`, planning stuck motions in this process so the statechart can run on the real robot; the demo running the cable demo once that way |
| `command_line.py`, `__main__.py` | the command line: `record`, `replan`, `summarize`, `joint-states`, `video`, `browse`, `trial` |
| `run_series.sh` | one complete series, as run for the results below |

The generic pieces live in giskardpy and cramph:
- `CloseToJointConfigurations`, a monitor (giskardpy)
- `Stalled`, a monitor turning True once a node stops approaching its goal (giskardpy)
- `RecentJointStates`, a monitor remembering the joint states of the last control cycles
  (giskardpy)
- `JointWaypointPath`, `RetractToLastSafeJointConfiguration` and
  `RetraceRecentJointStates`, tasks (giskardpy)
- `StoppedWhenTrue` and `HandledOnFailure`, goals (cramph), and `RecoveryByInsertion`, a
  failure handling strategy (coraplex)
- `CartesianPose.allow_replanning` (giskardpy), set to `False` by the cable actions for
  the motions in which both grippers hold the cable

## How a replanned motion runs

`ReplanningTransformation` is a coraplex plan transformation of `CartesianPose` goals.
`IllegalJointStateReplanning` replaces every pose goal of an arm's tool frame that
allows replanning and is a step of a sequence, in the actions that illegal joint states
were recorded in (`CableGraspAction`, `CableRegraspAction`, `CableRehangAction`), with:

```
HandledOnFailure  (cramph)
├── handled:  StoppedWhenTrue
│             ├── monitor:   CloseToJointConfigurations
│             │              observes True once the arms are within trigger_distance of
│             │              an illegal joint state, and remembers the last joint state
│             │              that was at least safe_distance from all of them
│             └── monitored: a copy of the pose goal, i.e. Giskard's optimization-based
│                            motion
└── strategy: RecoveryByInsertion, once: when the monitor stopped the goal, it inserts
              before it the Sequence "recovery of the stuck motion"
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
              and runs the goal again, from where the planned path left the arm
```

The recovery changes the statechart while it runs, so it only works in a statechart
ticked in this process, such as the kinematic simulation of a trial.

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

## Running a single experiment

There are two ways to run the demo once:

| | `trial` command | `stuck_motion_demo.py` |
|---|---|---|
| where | kinematic simulation in this process | simulated, standalone controller or real robot |
| workcell | perturbed by the seed (`TrialNoise`) | nominal, no noise |
| replanning | none, around illegal joint states, or when stuck (optionally retracting first) | when stuck, planned in this process |
| collisions recorded | yes (`ArmCollisionRecorder`) | no |
| video | optional | no |

First source ROS and the workspace in every terminal (see *Prerequisites*):

```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
cd <repository root>
```

### Simulated: one seeded trial (`trial`)

```bash
E="python -m experiments.daisy_cable_replanning"
D=/tmp/trial
# the demo as it is, Giskard without collision avoidance
$E trial --number 0 --seed 117 --directory $D/plain
# replanning around recorded illegal joint states
$E trial --number 0 --seed 117 --directory $D/guard \
    --illegal-joint-states <database> [<database> ...]
# collision avoidance; stuck motions are replanned
$E trial --number 0 --seed 117 --directory $D/stuck \
    --collision-avoidance --on-stagnation
# the same, but a stuck arm retracts to its last clear joint state first
$E trial --number 0 --seed 117 --directory $D/retract \
    --collision-avoidance --on-stagnation --retract-to last-clear-state
```

- `--video <file>.mp4` records the trial with MuJoCo. The file is written even when
  a stage fails.
- The trial exits with code 0 if every stage completed. Otherwise, the error is in
  `trial_NNN.errors.log`.
- The trial writes `trial_NNN.collisions.jsonl`, `trial_NNN.replanning.jsonl` and its
  logs into `--directory` (see *Output files*).
- To compare one seed with and without replanning side by side, use the `video`
  command with `--seed` (see *Step by step*).

### Simulated, standalone or real: the stuck-motion demo (`stuck_motion_demo.py`)

This runs the four stages once on the nominal workcell. Collision avoidance is on in
every stage but homing, and a stuck motion is replanned in this process (see *Running
it once, also on the real robot*). It doesn't retract before replanning.

**Simulated** needs nothing else running:

```bash
python -m experiments.daisy_cable_replanning.stuck_motion_demo --execution simulated
```

**Standalone:** Giskard simulates the joints and shows the robot in RViz. Terminal 1:

```bash
ros2 launch giskardpy_ros giskardpy_daisy_standalone.launch.py
```

Terminal 2, once RViz shows the robot:

```bash
python -m experiments.daisy_cable_replanning.stuck_motion_demo --execution standalone
```

**Real robot:**

> [!WARNING]
> Run the demo standalone first. Keep the emergency stop at hand: a
> planned path is followed with Giskard's collision avoidance off for the moving arm.

1. Bring up both UR5 arms and both Griplink grippers. The arm and gripper IP
   addresses are launch arguments (`left_robot_ip`, `right_robot_ip`,
   `left_gripper_ip`, `right_gripper_ip`); the launch file's defaults are DAiSy's:
   ```bash
   ros2 launch iai_daisy_description daisy_bringup.launch.py
   ```
2. Start Giskard on the arms' velocity controllers:
   ```bash
   ros2 launch giskardpy_ros giskardpy_daisy_velocity.launch.py
   ```
3. Run the demo:
   ```bash
   python -m experiments.daisy_cable_replanning.stuck_motion_demo --execution real
   ```

Every run appends the planned paths to `--replanning-log` (default
`stuck_motion_replanning.jsonl`).

## Repeating the experiment

### One command per series

```bash
cd <repository root>
experiments/src/experiments/daisy_cable_replanning/run_series.sh \
    experiments/results/daisy_cable_replanning/<series name> 3 \
    [databases of earlier series ...]
```

`TRIALS`, `WORKERS`, `MEMORY` and `HELDOUT_SEED` can be set as environment variables.
The script writes `summary.txt`, `joint_state_metrics.md` and `video/` into the
series directory. With 6 workers
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

# 4. numbers, joint state counts and video
$E summarize --output $S/heldout_replan --baseline $S/heldout_record
$E joint-states --series $S --rounds 2 [--extra-databases <databases of earlier series>]
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
| `collision_margin` | 1 mm | the planned path keeps the arm's bodies this far from the robot and the workcell (2 mm up to series 6) |
| `validity_checking_resolution` | 0.2 % | segment check spacing, as a fraction of the joint space's extent |
| `planning_time` | 10 s | RRT-Connect time limit per replanning |
| `waypoint_spacing` | 0.1 rad | largest joint step between waypoints handed to Giskard |
| `goal_seeds` | 64 | random inverse-kinematics seeds besides the current joint state |
| `goal_seed_spread` | π | largest joint offset of a seed |
| `retract_clearance` | 0 m | with `--retract-to last-clear-state`: distance on top of each pair's buffer zone that the retracted-to state keeps |
| `retraced_control_cycles` | 200 | with `--retract-to recent-steps`: how many of its latest control cycles a stuck arm retraces |
| `goal_sampling_seed` | 0 | seed of the seeds |
| `distinct_goal_distance` | 0.05 rad | goal configurations closer than this count as one |

## Output files

Every `record` or `replan` run writes into its output directory:

| file | content |
|---|---|
| `trial_results.json` | per trial: seed, failure (the error's last line, or none), number of illegal joint states, number of replanned motions |
| `illegal_joint_states.json` | the `IllegalJointStateDatabase` of the illegal joint states the trials ran into, without repeated poses (see below). Pass it to `--illegal-joint-states` |
| `trials/trial_NNN.collisions.jsonl` | one `IllegalJointState` per line: the 12 arm joint positions, the joints of the colliding arms, the collision kind, the two bodies, the penetration depth, the motion (`MotionLabel`: stage, index in the stage, action, motion type) |
| `trials/trial_NNN.replanning.jsonl` | one replanned motion per line: its label, where the arm retracted to, the planned waypoints |
| `trials/trial_NNN.log`, `.errors.log` | the trial process's output and error streams |

### Repeated poses

The trials of a run often collide in almost the same pose, e.g. the same forearm
contact in every trial. After all trials, `record` and `replan` therefore leave out
of `illegal_joint_states.json` every state that repeats an earlier pose:

- A repeat has the same colliding joints as an earlier state.
- On those joints, it differs from that state by at most `--duplicate-tolerance`
  (default 0.02 rad, the spacing the recorder logs states at; small against the
  guard's 0.15 rad trigger distance).
- The first state of each pose is kept.

The run prints how many states were left out. The trials' `.collisions.jsonl` logs
always keep every state.

| option | effect |
|---|---|
| (none) | repeated poses are left out, tolerance 0.02 rad |
| `--duplicate-tolerance 0.01` | a smaller (or larger) joint difference counts as the same pose |
| `--keep-duplicate-poses` | every logged state is written into the database |

For example, the series 2 recording holds 8,930 states. 3,547 of them are distinct
poses at 0.02 rad, 6,126 at 0.01 rad. There are no exact duplicates. The runs reported
below wrote their databases before this option existed, so they still hold every
state.

All JSON is written by `krrood.adapters.json_serializer`. `IllegalJointStateDatabase.load`
reads a database back.

### Joint state metrics

`joint-states` counts, for every run of a series (`joint_state_metrics.md`):

| column | meaning |
|---|---|
| states replanned around | states of all databases the run used |
| logged | states its trials logged (`trials/*.collisions.jsonl`) |
| duplicate share | share of the logged states that repeat an earlier pose at the duplicate tolerance |
| distinct poses | logged states left after removing the repeats |
| written | states of the database the run wrote |
| logged within guard reach | logged states within the trigger distance (0.15 rad) of a state the run replanned around. These are collisions the guard knew of and still did not prevent |

Below the table it reports three more numbers:
- the states of all training databases,
- the **coverage**: the share of the held-out baseline's collisions within the trigger
  distance of those databases, i.e. how many collisions of new seeds the guard would
  recognize,
- the time the guard's distance check takes per control cycle.

## Replanning when a motion gets stuck

A second trigger needs no database at all:
1. Giskard avoids collisions (`--collision-avoidance`, every stage but homing, as in
   the demo). With avoidance on, Giskard stops in front of a near collision instead of
   moving through it.
2. Every Cartesian motion runs with a stagnation monitor (`--on-stagnation`). Once the
   motion has gone `stagnation_timeout` (1.5 s of simulated time) without approaching
   its target, short of reaching it, OMPL plans a path to the same target, starting
   from where the arm got stuck.
3. Giskard follows that path, and the original plan continues with the next motion.

`StagnationReplanning` gives every pose goal that allows replanning this chart:

```
HandledOnFailure
├── handled:  StoppedWhenTrue
│             ├── monitor:   Stalled  (True once nothing short of its goal approached
│             │                        it for 1.5 s)
│             └── monitored: a copy of the pose goal
└── strategy: RecoveryByInsertion, inserting before the goal
              Sequence
              └── Parallel
                  ├── UpdateTemporaryCollisionRules  (Giskard's avoidance leaves the
                  │                                   arm's bodies alone while it runs)
                  └── PlannedJointPathToPose         (IK goals + RRT-Connect)
```

While the arm follows the planned path, Giskard's collision avoidance is off for the
arm's bodies. The planner has already checked the path against the bodies themselves,
with a 1 mm margin. Giskard's avoidance would keep the arm out of its 3 to 5 cm buffer
zones, which the path may pass through, and stop it again.

```bash
E="python -m experiments.daisy_cable_replanning"
$E record --output $S/baseline --collision-avoidance              # stuck without replanning
$E replan --output $S/replan_on_stagnation --collision-avoidance --on-stagnation
$E summarize --output $S/replan_on_stagnation --baseline $S/baseline
$E video --output $S/video --collision-avoidance --on-stagnation \
    --baseline $S/baseline --replanning $S/replan_on_stagnation
```

`series_4_replanning_on_stagnation/run.sh` runs both experiments.

### Retracting before replanning (`--retract-to`)

With `--retract-to <target>`, a stuck arm first moves back along the path it came, and
OMPL plans from where the retract ends. `RetractTarget` names the three targets:

| `--retract-to` | the arm retracts to | remembered by |
|---|---|---|
| `last-clear-state` | the last joint state of the motion in which every pair of the arm's bodies kept its Giskard buffer zone plus `retract_clearance`, or the motion's start if there was none | `PathSinceLastClearJointState` |
| `motion-start` | the joint state the motion started at, where the previous motion reached its goal | `PathSinceMotionStart` |
| `recent-steps` | the joint state the motion held `retraced_control_cycles` control cycles ago, or its start if it ran shorter | `RecentJointStates` (giskardpy) |

The first two remember the joint states the arm passed, at most `waypoint_spacing`
apart, and `RetractAlongRememberedPath` follows them backwards with Giskard's collision
avoidance on. `recent-steps` remembers the joint state of every control cycle, and
`RetraceRecentJointStates` (giskardpy) follows them backwards with the avoidance of the
arm off, like the planned path. It aims at the next remembered state once the arm is
within `waypoint_threshold` (0.05 rad) of the current one, and at most one state further
per control cycle, so the arm retraces the path it took instead of cutting its corners.
Either way the arm leaves the way it came, not on a straight line in joint space
(series 8 did that). The chart:

```
HandledOnFailure
├── handled:  StoppedWhenTrue
│             ├── monitor:   Stalled
│             └── monitored: Parallel (done once the pose goal is)
│                            ├── a copy of the pose goal
│                            └── PathSinceMotionStart, PathSinceLastClearJointState or
│                                RecentJointStates
└── strategy: RecoveryByInsertion, inserting before the goal
              Sequence
              ├── RetractAlongRememberedPath   (avoidance on), or
              │   RetraceRecentJointStates     (avoidance of the arm off)
              └── the planned path, as above
```

Series 8 and 9 ran with `--retract-when-stuck`, which is `--retract-to
last-clear-state` today.

### Running it once, also on the real robot (`stuck_motion_demo.py`)

`StagnationReplanning` changes the statechart while it runs, which only works in a
statechart ticked in this process. A plan for the real robot is sent to Giskard as one
statechart, so `StuckMotionReplanning` (`stuck_motion_replanning.py`) plans in this
process instead. Every pose goal that allows replanning becomes:

```
TryInOrder
├── StoppedWhenTrue
│   ├── monitor:   Stalled
│   └── monitored: a copy of the pose goal
└── JointPathToPose  (chooses its child in this process once it starts)
```

Once the goal is stopped, Giskard asks this process for the child of `JointPathToPose`.
`JointPathPlanning`, its child chooser, plans with `GoalConfigurationSampler` and
`JointSpacePlanner` from where the arm is in the synchronized world, and hands Giskard a
`FixedJointWaypointPath` (giskardpy) with the planned waypoints, with the arm's
collision avoidance off as above. Giskard has to import the `experiments` package to
read `JointPathToPose`.

The demo runs the four stages once on the nominal workcell, without trial noise.
Collision avoidance is on in every stage but homing. `--execution` takes `simulated`,
`standalone` (a standalone controller) or `real` (the real robot and its griplink
grippers). The motion labels in its replanning log are empty for runs against a
controller, because the motion tracker only sees the ticks of this process.

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

### Summary of all series

Every run of every series, counted from its `trial_results.json`. Each row counts
trials, not single collisions or motions.

- **Avoidance:** whether Giskard's collision avoidance was on. It is always off in the
  homing stage.
- **Failure detection:** how a motion that goes wrong is detected and handed to OMPL:
  - *none*: no replanning, the demo as it is;
  - *guard*: the arms come within `trigger_distance` of a recorded self-collision state;
  - *stuck*: the motion has not approached its target for `stagnation_timeout` (1.5 s of
    simulated time);
  - *stuck + retract*: as *stuck*, but the arm first retraces its path, to its last
    joint state outside Giskard's buffer zones or to where the motion started
    (`--retract-to`).
- **Completed:** the trial performed all four stages and its process exited without an
  error.
- **Succeeded:** completed *and* never touched itself. This is the success rate.
- **Colliding:** at least one self-collision was recorded, whether the trial completed or
  not. So *completed* minus *succeeded* trials completed despite a collision, and
  *colliding* minus that many collided and then failed.

| series (what changed) | run | trials (seeds) | avoidance | failure detection | completed | succeeded | colliding |
|---|---|---|---|---|---|---|---|
| **1** safe postures | `record` | 100 (0-99) | off | none | 94 | 31 | 63 |
| | `replan` | 100 (0-99) | off | guard | 94 | 94 | 0 |
| | `heldout_record` | 50 (100-149) | off | none | 44 | 14 | 30 |
| | `heldout_replan` | 50 (100-149) | off | guard | 44 | **44** | 0 |
| **2** no safe postures | `record` | 100 (0-99) | off | none | 79 | 0 | 94 |
| | `replan_round_1` | 100 (0-99) | off | guard | 70 | 0 | 83 |
| | `replan_round_2` | 100 (0-99) | off | guard | 58 | 40 | 21 |
| | `replan_round_3` | 100 (0-99) | off | guard | 20 | 17 | 12 |
| | `heldout_record` | 100 (100-199) | off | none | 76 | 0 | 87 |
| | `heldout_replan` | 100 (100-199) | off | guard | 61 | **3** | 60 |
| | `heldout_replan_any_goal` | 100 (100-199) | off | guard | 54 | 17 | 58 |
| | `heldout_replan_unseeded` | 100 (100-199) | off | guard | 60 | 3 | 61 |
| **3** duplicate poses removed | `record` | 100 (0-99) | off | none | 79 | 0 | 94 |
| | `replan_round_1` | 100 (0-99) | off | guard | 72 | 0 | 82 |
| | `replan_round_2` | 100 (0-99) | off | guard | 60 | 47 | 15 |
| | `replan_round_3` | 100 (0-99) | off | guard | 68 | 38 | 30 |
| | `heldout_record` | 100 (100-199) | off | none | 76 | 0 | 87 |
| | `heldout_replan` | 100 (100-199) | off | guard | 28 | **26** | 37 |
| | `heldout_replan_all_states` | 100 (100-199) | off | guard | 29 | 24 | 40 |
| **4** stuck trigger (4 mm gripper shapes) | `baseline` | 100 (0-99) | on | none | 0 | 0 | 0 |
| | `replan_on_stagnation` | 100 (0-99) | on | stuck | 97 | **94** | 3 |
| **5** full-size gripper shapes | `record` | 100 (0-99) | off | none | 79 | 0 | 94 |
| | `replan_round_1` | 100 (0-99) | off | guard | 70 | 0 | 83 |
| | `replan_round_2` | 100 (0-99) | off | guard | 94 | 0 | 94 |
| | `replan_round_3` | 100 (0-99) | off | guard | 91 | 15 | 76 |
| | `heldout_record` | 100 (100-199) | off | none | 76 | 0 | 87 |
| | `heldout_replan` | 100 (100-199) | off | guard | 73 | **67** | 8 |
| **6** as 4, full-size shapes | `baseline` | 1000 (0-999) | on | none | 0 | 0 | 0 |
| | `replan_on_stagnation` | 1000 (0-999) | on | stuck | 0 | **0** | 0 |
| **7** 1 mm margin, targets set back | `baseline` | 1000 (0-999) | on | none | 0 | 0 | 0 |
| | `replan_on_stagnation` | 1000 (0-999) | on | stuck | 540 | **471** | 72 |
| **8** retract first, straight line | `replan_retract_first` | 1000 (0-999) | on | stuck + retract to last clear state | 502 | **500** | 5 |
| **9** retract along the path, cable-safe flags, 5 mm forearm buffer | `baseline` | 1000 (0-999) | on | none | 0 | 0 | 0 |
| | `replan_retract_first` | 1000 (0-999) | on | stuck + retract to last clear state | 0 | **0** | 39 |
| **10** 2 mm buffer to post and hanger | `baseline` | 1000 (0-999) | on | none | 622 | **622** | 0 |
| | `replan_retract_first` | 1000 (0-999) | on | stuck + retract to last clear state | 0 | 0 | 33 |
| **11** retract to the motion's start | `replan_retract_to_motion_start` | 1000 (0-999) | on | stuck + retract to motion start | 340 | **286** | 90 |

Series 8 and 11 have no baseline row: they reuse the baseline of series 7 and 10.
Trials without replanning reproduce exactly from their seed, so the baseline is valid
for both.

How to read it:

- **Guard trigger, avoidance off (series 1-3, 5).** The baselines complete often but
  nearly always collide: 0 of 100 succeed in series 2, 3 and 5. The best held-out result
  is series 5, with 67 of 100.
- **Stuck trigger, avoidance on (series 4-11).**
  - Series 4's 94 of 100 relied on gripper collision shapes that were only about 4 mm
    large, because of a mesh-scaling bug.
  - With full-size shapes, targets set back and a 1 mm planner margin, series 7 reaches
    471 and series 8 reaches 500. Both replan motions while both grippers hold the cable,
    which can damage it.
  - With that ruled out (series 9), no trial succeeds.
  - The 2 mm buffer to the post and hanger (series 10) lets 622 of 1000 succeed *without*
    replanning.
  - The best replanning variant on that setup is series 11, with 286 of 1000, still well
    below series 10's baseline.
- **Run sizes differ.** Series 1-5 ran 100 trials per run (50 for the held-out runs of
  series 1), series 6-11 ran 1000.
- **Rows are only comparable within a series.** The code, the collision shapes and the
  buffer zones changed between series.

How failures and collisions are detected:

- **A trial fails** when its process raises an error:
  - Giskard makes no progress for 5 s (`MotionMadeNoProgress`);
  - the planner finds no valid goal (`NoValidGoalConfiguration`) or no path
    (`NoJointPathFound`);
  - collision avoidance is violated (`MotionViolatedCollisionAvoidance`);
  - a motion exceeds 120 s of simulated time.
- **A collision** is measured separately by the recorder in every trial, with or without
  avoidance, whenever a body of an arm touches another body of the robot. That includes
  the other arm, the arm's own parts and the table. It does not include the cable, the
  post, the hanger, or near misses inside a buffer zone.

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

### Series 3: series 2 repeated with repeated poses removed

Directory: `series_3_duplicate_poses_removed/`.

Series 3 is `run_series.sh` with 3 rounds and series 1's database as an extra input,
exactly as series 2 but with the current code: duplicate removal at its default (0.02
rad), 64 goal seeds, nearest goal first, seeded OMPL. The runs without replanning
reproduce series 2 exactly (8,930 and 8,260 logged states).

Joint state metrics (`joint_state_metrics.md`):

| run | states replanned around | logged | duplicate share | distinct poses | written | logged within guard reach |
|---|---|---|---|---|---|---|
| record | 0 | 8,930 | 60% | 3,547 | 3,547 | 0 |
| heldout_record | 0 | 8,260 | 59% | 3,397 | 3,397 | 0 |
| replan_round_1 | 4,996 | 3,264 | 43% | 1,874 | 1,874 | 28 |
| replan_round_2 | 6,870 | 287 | 55% | 129 | 129 | 0 |
| replan_round_3 | 6,999 | 4,582 | 51% | 2,223 | 2,223 | 0 |
| heldout_replan | 9,222 | 2,261 | 56% | 995 | 995 | 0 |

- The training databases hold 9,222 states; series 2's held 23,380.
- They still cover 100 % of the held-out baseline's collisions, as series 2's did.

Outcomes (`summary.txt`):

| run | completed | trials colliding | completed without any collision |
|---|---|---|---|
| `record` / `heldout_record` | 79 / 76 | 94 / 87 | 0 / 0 |
| `replan_round_1` | 72 | 82 | 0 |
| `replan_round_2` | 60 | 15 | 47 |
| `replan_round_3` | 68 | 30 | 38 |
| `heldout_replan` | 28 | 37 | 26 |

**Effect of removing repeated poses.** Series 2 and 3 cannot be compared directly,
because their training rounds ran with different code. The feature was therefore
measured on identical training data: the held-out replanning was repeated with
databases of *all* logged states of the same series 3 runs
(`all_logged_states/`, `heldout_replan_all_states/`).

| held-out seeds 100-199 | repeated poses removed | all logged states |
|---|---|---|
| states in the training databases | 9,222 | 18,512 |
| size of the database files | 10.0 MB | 20.1 MB |
| guard check per control cycle | 0.27 ms | 0.59 ms |
| planning time per replanned motion, mean / median | 0.53 s / 0.24 s | 1.00 s / 0.46 s |
| completed | 28 | 29 |
| trials colliding | 37 | 40 |
| completed without any collision | 26 | 24 |
| illegal joint states logged | 2,261 | 2,464 |
| replanned motions | 509 | 505 |

Removing repeated poses halves the databases, and with them the guard's check per
control cycle and the planning time. The outcomes stay the same within the run-to-run
variation of replanning.

**Other findings.**
- **Guard reach.** Collisions the guard knew of are rare: at most 28 logged states per
  run lie within the trigger distance of the databases used. Nearly all collisions
  after replanning happen in poses no earlier run has visited. The database
  limits what the guard can prevent, not the guard itself.
- **Rounds do not converge steadily.** Round 3 logs more collisions than round 2 (4,582
  vs 287). Each round's replans lead the arms into new configurations, and those can
  collide anew.
- **Completion.** Compared with series 2, more held-out runs complete without any
  collision (26 vs 3), but fewer complete at all (28 vs 61). 31 of them stall in a
  Cartesian motion of the rehang (`MoveTCP/position#18`), right after a replanned
  regrasp. The run with all logged
  states stalls there just as often (30), so the stall comes from series 3's training
  history, not from removing repeated poses.

The video `series_3_duplicate_poses_removed/video/daisy_cable_replanning_seed102.mp4`
shows held-out seed 102.

### Series 4: replanning when a motion gets stuck

Directory: `series_4_replanning_on_stagnation/`, run by its `run.sh`. It uses seeds 0-99
and the demo without safe intermediate postures. Giskard avoids collisions in both
runs, and no database is used.

| seeds 0-99 | Giskard only (series 3 `record`) | collision avoidance | collision avoidance + OMPL when stuck |
|---|---|---|---|
| completed | 79 | 0 | 97 |
| completed without any collision | 0 | 0 | 94 |
| trials colliding | 94 | 0 | 3 |
| illegal joint states logged | 8,930 | 0 | 7 |
| replanned motions | - | - | 660 (6.6 per trial) |
| failures | 21 stalls | 100 stalls (`MotionMadeNoProgress`) | 3 over the 120 s simulated-time limit |

- **Without replanning,** collision avoidance prevents every collision, but Giskard
  gets stuck in front of the near collisions: 94 trials in the same Cartesian motion of
  the grasp, 5 in another. Not one trial completes.
- **With replanning,** the stuck motions are replaced by planned paths, and 97 trials
  perform the whole demo. RRT-Connect found every path, in 18 ms median and 0.6 s at
  most; without an illegal joint state database there is less to check.
- **The 7 remaining contacts** are grazes of at most 0.4 mm, all while the arm follows
  a planned path with Giskard's avoidance off for it. Giskard moves each joint towards
  the next waypoint at its own speed, so the executed path deviates slightly from the
  straight segment the planner checked; the 2 mm margin covers that almost always.
- **The 3 failures** are planned paths followed too slowly to finish within coraplex's
  120 s of simulated time per motion.

`series_4_replanning_on_stagnation/video/daisy_cable_replanning_seed0.mp4`, also in the
repository's `video/` directory, shows seed 0:
- Left, Giskard with collision avoidance: stuck in the grasp.
- Right, with replanning when stuck: 8 motions replanned, no collision, every stage
  completed.

### Series 5: series 3 repeated with full-size gripper collision shapes

Directory: `series_5_fixed_collision_shapes/`.

Until commit `4739a9995`, Bullet scaled the convex decomposition of a mesh twice.
DAiSy's gripper meshes (scale 0.001) became about 4 mm large, so neither the recorder,
the planner nor Giskard saw the gripper base, mounting plate and fingers. Series 1 to 4
ran with these shapes. Series 5 is `run_series.sh` with 3 rounds, as series 3, but
without series 1's database, so every state comes from the full-size shapes.

The runs without replanning collide in the same trials as series 3 (94 and 87), but in
more states, and at the gripper bodies the bug had shrunk:

| colliding pair, states logged | series 3 `record` | series 5 `record` | series 3 `heldout_record` | series 5 `heldout_record` |
|---|---|---|---|---|
| right forearm / right gripper base | 2,206 | 5,441 | 2,283 | 5,676 |
| right forearm / right gripper side cylinder | 5,180 | 4,011 | 4,353 | 3,093 |
| right forearm / right finger tips | 1,208 | 401 | 1,177 | 416 |
| right forearm / *left* gripper base | 0 | 176 | 0 | 208 |
| other pairs | 336 | 219 | 447 | 304 |
| all | 8,930 | 10,248 | 8,260 | 9,697 |

The recorder logs one pair per state, so a state now counted at the gripper base was
often counted at the side cylinder or a finger tip before.

Joint state metrics (`joint_state_metrics.md`):

| run | states replanned around | logged | duplicate share | distinct poses | written | logged within guard reach |
|---|---|---|---|---|---|---|
| record | 0 | 10,248 | 56% | 4,507 | 4,507 | 0 |
| heldout_record | 0 | 9,697 | 57% | 4,189 | 4,189 | 0 |
| replan_round_1 | 4,507 | 966 | 57% | 418 | 418 | 28 |
| replan_round_2 | 4,925 | 20,837 | 71% | 5,958 | 5,958 | 0 |
| replan_round_3 | 10,883 | 13,267 | 59% | 5,499 | 5,499 | 1 |
| heldout_replan | 16,382 | 17 | 0% | 17 | 17 | 0 |

The training databases hold 16,382 states (series 3: 9,222) and cover 100 % of the
held-out baseline's collisions. The guard's check takes 0.52 ms per control cycle.

Outcomes (`summary.txt`), series 3 / series 5:

| run | completed | trials colliding | completed without any collision | replanned motions |
|---|---|---|---|---|
| `record` | 79 / 79 | 94 / 94 | 0 / 0 | - |
| `heldout_record` | 76 / 76 | 87 / 87 | 0 / 0 | - |
| `replan_round_1` | 72 / 70 | 82 / 83 | 0 / 0 | 287 / 314 |
| `replan_round_2` | 60 / 94 | 15 / 94 | 47 / 0 | 489 / 159 |
| `replan_round_3` | 68 / 91 | 30 / 76 | 38 / 15 | 533 / 388 |
| `heldout_replan` | 28 / 73 | 37 / 8 | 26 / 67 | 509 / 466 |

- **Held-out seeds.** With series 5's databases, 67 of 100 held-out trials complete
  without any collision (series 3: 26). 73 complete at all (28), only 8 collide (37),
  and they log 17 illegal joint states (2,261). Series 3's stall in
  `MoveTCP/position#18`, which ended 31 of its held-out runs, does not occur. Of the
  27 failures, 14 are stalls
  (`MotionMadeNoProgress`, 12 of them in `MoveTCP/position#15`), 11 find no path
  (`NoJointPathFound`), and 2 exceed the simulated-time limit.
- **Training rounds.** Round 2 collides in 94 trials and logs 20,837 states; 928 of
  the 5,958 it writes are the right upper arm against the table. Round 3 still collides in 76. The rounds again do not converge
  steadily, but together their states make the held-out run nearly collision-free.
- **Remaining held-out contacts**, in logged states: the left wrist against a right
  finger tip (6), the right forearm against its gripper's wrist cylinder (6), the right
  forearm against its gripper base (2), and single contacts between the wrists.
- **What else changed since series 3.** Besides the collision shapes and the missing
  series 1 database, the code gained `300fd746f` (`AllowCollisionForBodies` keeps its
  bodies) and `68398828a` (the planner takes the arm's pairs from the standing rules
  only). The runs without replanning are unaffected by those, the replanning runs may
  not be. Replanning also reproduces only statistically, and this is one run.

`series_5_fixed_collision_shapes/video/daisy_cable_replanning_seed100.mp4` shows
held-out seed 100:
- Left, Giskard alone: 68 illegal joint states.
- Right, with replanning around series 5's databases: 7 motions replanned, no
  collision, every stage completed.

### Series 6: series 4 with full-size gripper collision shapes, 1000 trials

Directory: `series_6_stagnation_full_size_meshes/`, run by its `run.sh` at commit
`0dfd08c3a`. It is series 4 (collision avoidance, no database) with seeds 0-999. The
gripper meshes are scaled only once (`8e85d0f57e`), and convex hulls load without a
margin (`417079692`).

| seeds 0-999 | collision avoidance | collision avoidance + OMPL when stuck |
|---|---|---|
| completed | 0 | 0 |
| completed without any collision | 0 | 0 |
| trials colliding | 0 | 0 |
| replanned motions | - | 1,481 (1 to 4 per trial) |
| failures | 1,000 stalls (`MotionMadeNoProgress`) | 1,000 `NoValidGoalConfiguration` |

- **Without replanning,** every trial stalls in the grasp: 916 in
  `MoveTCP/position#18`, 80 in `MoveTCP/position#14`, 4 in a `SequentialNode`. Series 4
  stalled there too.
- **With replanning,** the stuck motions are replanned, and the arm gets further into
  the grasp, but every trial ends in a motion whose *target pose itself* is too close
  to the workcell. Every inverse-kinematics goal configuration has the right finger
  tip
  - 0.1 to 0.5 mm from the post's item profile (740 trials), or
  - 2.5 mm inside the cable hanger (260 trials).

  Both are below the planner's 2 mm `collision_margin`, so no path is searched.
  Before the fix, the gripper shapes were about 4 mm large, so these contacts didn't
  exist and series 4 completed 97 trials.
- **Collision avoidance with the full-size shapes works.** `closeups/probe_stuck.py`
  reruns seeds 0 to 3 in both conditions. It tracks the smallest distance of every arm
  body to the robot and the workcell over the whole run. The cable is left out, and so
  are contacts the recorder doesn't count.
  - The arms never come closer than 15 mm to anything.
  - Giskard stops the finger tips 29 to 30 mm in front of the item profile and 28 mm
    in front of the hanger.
  - The close-ups in `closeups/` show the visible gap.
- **The recorded failure** is the last line of the error stream, here the last goal
  configuration of the `NoValidGoalConfiguration` message, not the exception's name.

`series_6_stagnation_full_size_meshes/video/daisy_cable_replanning_seed0.mp4` shows
seed 0:
- Left, Giskard with collision avoidance: stuck in front of the post in the grasp.
- Right, with replanning when stuck: 3 motions replanned, then stopped in front of the
  hanger, whose target the planner rejects.

### Series 7: series 6 with a 1 mm margin and set-back targets

Directory: `series_7_stagnation_targets_set_back/`, run by its `run.sh`. It is series 6,
with seeds 0-999, plus two fixes for the targets the planner rejected in series 6:

1. **A 1 mm planner margin.** `ReplanningParameters.collision_margin` went from 2 mm
   to 1 mm. This alone fixes neither target: the scoop target leaves 0.1 to 0.5 mm to
   the post, and the free-cable target reaches 2.5 mm into the hanger.
2. **Set-back targets.** `CableGraspAction` moves two targets of the scoop arm back,
   against their approach:
   - `scoop_setback` (1 mm): the scoop target, away from the post. The finger tip
     then keeps 1.1 mm.
   - `free_cable_setback` (3.5 mm): the free-cable target, diagonally back and down,
     along the contact normal at the hanger's front edge. The finger tip then keeps
     1.0 to 1.3 mm. A 1 mm setback would still leave it 1.5 mm inside the hanger.

   This changes the demo on the real robot too.

| seeds 0-999 | collision avoidance | collision avoidance + OMPL when stuck |
|---|---|---|
| completed | 0 | 540 |
| completed without any collision | 0 | 471 |
| trials colliding | 0 | 72 |
| illegal joint states logged | 0 | 523 |
| replanned motions | - | 6,973 (7 per trial) |

- **Without replanning,** every trial still stalls in the grasp, as in series 6.
- **Remaining failures with replanning (460):**
  - 346 stalls in the regrasp: 336 in `MoveTCP/position#13` (47 of them together
    with `MoveTCP/orientation#14`) and 10 in `MoveTCP/position#15`. A replannable
    motion hands over to the planner after 1.5 s without progress, before the 5 s
    stall limit, so these are motions of `CableRegraspAction` created with
    `allow_replanning=False`.
  - 110 motions that didn't end within 120 s of simulated time.
  - 4 stalls while following a planned path.
- **Remaining collisions** are self-contacts of the right arm, the forearm against its
  own gripper (side cylinder, base, wrist cylinder), almost all in `rehang#1` while it
  follows a planned path with Giskard's avoidance off. They are shallow: median
  0.7 mm, at most 4.6 mm. Giskard follows the waypoints at its own joint speeds, so
  the executed path leaves the segments the planner checked by more than the 1 mm
  margin. 69 of the 72 colliding trials still complete.

`series_7_stagnation_targets_set_back/video/daisy_cable_replanning_seed0.mp4` shows
seed 0:
- Left, Giskard with collision avoidance: stuck in front of the post in the grasp.
- Right, with replanning when stuck: 10 motions replanned, no collision, every stage
  completed.

### Series 8: series 7 retracting before replanning

Directory: `series_8_stagnation_retract_first/`, run by its `run.sh`. It is series 7's
replanning run, seeds 0-999, with `--retract-when-stuck`: a stuck arm first moves back
to its last joint state outside Giskard's buffer zones (`retract_clearance` 0), and
OMPL plans from there. The baseline without replanning is series 7's: trials without
replanning reproduce exactly, and their code did not change.

`retract_clearance` was chosen on seeds 0-47:

| seeds 0-47 | completed | completed without any collision | colliding |
|---|---|---|---|
| series 7, no retract | 25 | 22 | 3 |
| retract, buffer zone + 1 cm | 2 | 2 | 4 |
| retract, buffer zone + 0 | 23 | 23 | 0 |

With 1 cm on top, no state of a motion is clear in most cases, so the arm retracts
up to 1.8 rad to where the motion started, and the trials then stall or time out.

| seeds 0-999 | series 7, no retract | series 8, retract first | McNemar p |
|---|---|---|---|
| completed | 540 | 502 | 0.004 |
| completed without any collision | 471 | 500 | 0.06 |
| trials colliding | 72 | 5 | < 1e-16 |
| illegal joint states logged | 523 | 7 | |
| replanned motions | 6,973 | 7,061 | |

The McNemar test compares the two runs seed by seed.

- **Collisions almost disappear.** The forearm grazing its own gripper in `rehang#1`
  (series 7) is gone. 5 trials still collide, in 7 states, at the gripper base and
  between the wrists.
- **Success rises slightly,** from 47.1 % to 50.0 %. With p = 0.06, that difference
  could still be chance.
- **Completion falls,** from 54.0 % to 50.2 %, because the retract itself can get
  stuck. In 98 trials, a retract to the last clear state stops approaching its target
  (`MotionMadeNoProgress`). The retract runs with Giskard's collision avoidance on and
  moves straight in joint space, not back along the path the arm came.
- **Remaining failures:**
  - 286 stalls in the regrasp, in motions created with `allow_replanning=False`
    (series 7: 346)
  - 109 motions over the 120 s simulated-time limit (110)
  - 5 stalls while following a planned path

`series_8_stagnation_retract_first/video/daisy_cable_replanning_seed0.mp4` shows
seed 0:
- Left, Giskard with collision avoidance: stuck in the grasp.
- Right, retracting before replanning: 10 motions replanned, no collision, every stage
  completed.

### Series 9: retracting along the path, cable-safe motions, 5 mm forearm buffer

Directory: `series_9_retract_along_path/`, run by its `run.sh`. Seeds 0-999, collision
avoidance, without and with replanning (retracting first). Compared with series 8,
three things changed:

1. **The retract follows the arm's path back** (`PathSinceLastClearJointState`), not
   a straight line in joint space.
2. **No replanning while both grippers hold the cable.** A planned joint path changes
   the distance between the grippers along the way, which can damage the cable. The
   five motions of `CableRegraspAction` marked `allow_replanning=False` all run with
   both grippers closed on the cable, so each flag is needed. Two more such motions
   now have it:
   - in `CableGraspAction`, the scoop arm's move back to `pre_free_cable_pose` after
     closing on the cable at the hanger;
   - in `CableRegraspAction`, the free arm's move to `inter_free_arm_pose`.

   The scoop arm's move to `clear_scoop_pose` and its quarter turn there also run with
   both grippers closed, but may be replanned. Series 9 ran with them flagged too.
   Tests in `test_replanning.py` check which motions are not replanned.
3. **5 mm buffer between a forearm and its own gripper.** In `DAiSy`, all bodies of a
   gripper (`DAiSyLink`) get `forearm_to_own_gripper_buffer_zone_distance` (5 mm) to
   the forearm of the same arm. Before, it was 15 mm for the cylinders and 30 mm for
   the rest. This changes collision avoidance on the real robot too.

| seeds 0-999 | collision avoidance | + retract and replan when stuck | series 8 |
|---|---|---|---|
| completed | 0 | **0** | 502 |
| trials colliding | 0 | 39 | 5 |
| replanned motions | - | 3,957 | 7,061 |

- **No trial completes.** 716 of them stall in `regrasp#9`, the free arm's move to
  `inter_free_arm_pose`. For seed 0, the inverse-kinematics solution nearest to where
  the arm is puts the right forearm 39 mm *into* its own gripper base. The only
  collision-free solutions are 3 rad or more away, i.e. the arm has to turn over.
  - In series 7 and 8, `regrasp#9` was replanned in about 700 trials, and OMPL did
    exactly that, with both grippers on the cable. Their success rates depended on
    replans that the cable doesn't allow.
  - The regrasp can only complete without such a replan if `inter_free_arm_pose` is
    reachable from where the free arm grasps the cable.
- **The 39 colliding trials** end in `regrasp#9` too, with
  `MotionViolatedCollisionAvoidance`: with only 5 mm of buffer, Giskard lets the
  forearm touch its own gripper base (up to 2.3 mm deep) on the way to that target.
- **The other stalls:**
  - 234 in `MoveTCP/position#13` (series 8: 276). In seed 2 this is `regrasp#6`,
    the free arm sliding along the cable, now held back by its upper arm 3 cm from the
    post;
  - 10 in `MoveTCP/position#15`.
- **The retract no longer gets stuck**: none of the 1000 trials stalls in it (series 8:
  98).
- The recorded failure of the 39 colliding trials is empty, because their error log
  ends in a blank line.

The whole-demo tests (`test_replanning_on_stagnation_completes_the_demo_without_collision`,
`test_retracting_before_replanning_completes_the_demo_without_collision`,
`test_simulated_demo_completes_by_replanning_stuck_motions`) fail for the same reason.

### Why the baseline gets stuck

The trials without replanning (collision avoidance on) all stall in the grasp:
- **916 of 1000 in the scoop** (`grasp#5`, `MoveTCP/position#18`). The scoop target is
  1.1 mm from the post, to put the fingers around the cable. Giskard keeps the finger
  tips 3 cm from the post, because the post hangs below the robot's root (the table),
  so the robot's 3 cm self-collision buffer applies. It stops about 29 mm in front of
  the post and never reaches the target.
  - Before the mesh fix (series 1 to 5), the finger tips were 4 mm blobs, which
    collision avoidance did not see.
  - The demo script itself runs without collision avoidance.
- **80 in the approach** (`grasp#3`, `MoveTCP/position#14`). From the parked posture,
  the approach turns `right_wrist_2_joint` from 1.56 rad to 0, the UR wrist
  singularity, and the Cartesian motion stops making progress there. This happens only
  for post positions whose approach path crosses it.
- 4 in a `SequentialNode`.

### Series 10: a 2 mm buffer between the arms and the post and hanger

Directory: `series_10_post_buffer_2mm/`, run by its `run.sh` at commit `bb04e1d76`.
The demo with collision avoidance, seeds 0-999, without replanning and with replanning
when stuck (retracting along the path first).

`CableWorkcell.find_or_add_to` (the demo script) adds a collision rule: every body an
arm moves keeps `arm_buffer_zone_distance` (2 mm) from the post and the hanger.
Before, the robot's 3 cm self-collision buffer applied, because the post is mounted
under the robot's root. All other buffer zones stay as they are.

The distance was chosen on the nominal workcell with a robot-wide self buffer:
- 3 cm, 2 cm, 1 cm, 5 mm, 4 mm and 3 mm all stall in the scoop.
- 2 mm completes the demo.

The scoop target is 1.15 mm from the post, and the motion counts as done within 1 mm
of it. A rule for the grippers alone was not enough: the rehang then stalls with the
forearm 30 mm in front of the hanger.

| seeds 0-999 | 3 cm, no replanning (series 9) | 2 mm, no replanning | 2 mm, retract and replan |
|---|---|---|---|
| completed | 0 | 622 | **0** |
| trials colliding | 0 | 0 | 33 |
| replanned motions | - | - | 2,031 |

The 378 remaining failures are all stalls (`MotionMadeNoProgress`):
- 80 in the approach of the grasp (`MoveTCP/position#14`), at the wrist singularity
  (`right_wrist_2_joint` near 0);
- the rest in the regrasp. In seeds 2 and 3, the right forearm stops 3 to 4 mm from
  its own gripper, inside the 5 mm buffer between a forearm and its own gripper.
  Without avoidance, these motions drive the forearm into the gripper.

**Replanning makes it worse.** With replanning, 958 trials stall in `regrasp#9`, the
free arm's move to `inter_free_arm_pose`, which holds the cable with both grippers
and is not replanned. The cause is the earlier replan of `regrasp#3`, the free arm's
move in front of the cable (893 trials):
- The stagnation monitor hands a motion to the planner after 1.5 s without progress.
  Without replanning, Giskard is still allowed 5 s and then finishes these motions.
- The planned path ends in a different inverse-kinematics solution for the same
  pose. In seed 0, the right shoulder pan is at 2.9 rad after the replanned
  `regrasp#3`, against 0.54 rad without replanning.
- From that configuration, Giskard can't reach the target of `regrasp#9`.

`grasp#3`, the approach of the scoop arm, is replanned in 999 trials as well. It ends
in practically the same configuration (within 0.08 rad in seed 0) and does no harm.
The 33 colliding trials touch the right gripper base with the forearm (at most
0.01 mm) in `regrasp#9` and abort with `MotionViolatedCollisionAvoidance`.

`series_10_post_buffer_2mm/video/daisy_cable_replanning_seed0.mp4` shows seed 0:
- Left, Giskard with collision avoidance: every stage completed.
- Right, retracting to the last clear state and replanning: after the replanned
  `regrasp#3`, the right arm is turned elbow-up and stalls in `regrasp#9`.

### Series 11: series 10 retracting to the motion's start

Directory: `series_11_retract_to_motion_start/`, run by its `run.sh`. Series 10's
replanning run with `--retract-to motion-start`: a stuck arm retraces its path to
where the motion started, the goal the previous motion reached, and OMPL plans from
there. The baseline is series 10's.

| seeds 0-999 | no replanning | retract to last clear state (series 10) | retract to motion start |
|---|---|---|---|
| completed | 622 | 0 | 340 |
| completed without any collision | 622 | 0 | 286 |
| trials colliding | 0 | 33 | 90 |
| replanned motions | - | 2,031 | 1,978 |

Retracting to the motion's start keeps the arm in the configuration Giskard reached,
so 340 trials get through `regrasp#9`. Without replanning, 622 do. 48 seeds complete
only with replanning, 340 only without it (McNemar p < 1e-48).

Failures with replanning (660):
- 247 stall in `regrasp#9` (`MoveTCP/position#19`), the cable-holding motion that is
  not replanned.
- 204 end with `NoValidGoalConfiguration`, all still in the grasp, mostly in `grasp#4`
  (replanned 362 times). From the motion's start, every goal configuration found puts
  the right forearm into its gripper's side cylinder or the upper arm into the table.
- 156 end with `MotionViolatedCollisionAvoidance` while following a planned path, with
  the forearm in its own gripper's side or wrist cylinder: 276 logged states, median
  0.3 mm, up to 18 mm deep. Their recorded failure is empty.
- 38 stall in the retract or the planned path, 9 in `MoveTCP/position#13`, 6 exceed
  the simulated-time limit.

`series_11_retract_to_motion_start/video/daisy_cable_replanning_seed0.mp4` shows seed 0:
- Left, Giskard with collision avoidance: every stage completed.
- Right, retracting to the motion's start and replanning: 3 motions replanned, no
  collision, every stage completed.

## Known limitations

- The guard is joint-space proximity to *recorded* states. It does not foresee
  collisions in regions no trial has visited; a distance-based monitor on the collision
  checker would.
- `plan_waypoints` (inverse kinematics plus planning) runs inside one control cycle.
  That is fine in simulation, but on the real robot it would stall the controller for
  up to several seconds.
- `AllowCollisionForBodies` (semantic_digital_twin) used to forget its bodies the
  first time the collision manager updated it. This made the demo's rule allowing
  every collision of the cable ineffective, and it was fixed for series 4. Series 1 to
  3 ran before the fix, so their collision matrices still contained cable pairs; the
  recorder and the planner never counted the cable, so their numbers are unaffected.
- The cable is ignored by the planner. It is soft and meant to touch the hanger and
  the grippers.
- In the videos, the cable stays on the post while the grippers carry it. The MuJoCo
  mirror does not follow reattachments; the simulated world does.
- The JSON files of series 1 originally stored the stage of a `MotionLabel` as an
  enum of a module that no longer exists. They were migrated to the plain stage name.
