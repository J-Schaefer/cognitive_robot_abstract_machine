#! /usr/bin/env python3
"""
DAiSy grasps a cable from its hanger with both arms, regrasps it above the table and
hangs it back onto the hanger.

Choose where the plan runs with :data:`executor_type`:

- :class:`~coraplex.plans.executors.KinematicallySimulatedPlanExecutor` simulates DAiSy
  in this process,
- :class:`~coraplex.plans.executors.RealRobotPlanExecutor` sends the plan to a running
  standalone controller (``ros2 launch giskardpy_ros giskardpy_daisy_standalone.launch.py``),
- :class:`~coraplex.griplink_grippers.RealDAiSyPlanExecutor` drives the real DAiSy and
  its griplink grippers.

Every phase is performed on its own, since each cable action computes its poses from the
world as the previous phase left it.
"""

# %% Imports

from __future__ import annotations

from math import pi
from pathlib import Path
from time import sleep

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.griplink_grippers import RealDAiSyPlanExecutor
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
    SimulatedPlanExecutor,
)
from coraplex.robot_plans.actions.core.cable_actions import (
    CableGraspAction,
    CableRegraspAction,
    CableRehangAction,
)
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from typing_extensions import Callable
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelReloadSynchronizer,
    WorldSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.griplink_gripper import GriplinkFlexConfiguration
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.cable import Cable
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

verbose = True
collision_avoidance = True
executor_type: type[PlanExecutor] = RealRobotPlanExecutor
# executor_type = RealDAiSyPlanExecutor
# executor_type = KinematicallySimulatedPlanExecutor

print(f"Running with: {executor_type.__name__}")

objects_directory = Path(__file__).resolve().parents[2] / "resources" / "objects"

# %% Robot and World Setup
ros_session = RobotDemonstrationRosSession.start("daisy_cable_demo_node")
if issubclass(executor_type, SimulatedPlanExecutor):
    world = URDFParser.from_file(file_path=DAiSy.get_ros_file_path()).create_world()
    DAiSy.from_world(world)
    VizMarkerPublisher(_world=world, node=ros_session.node)
else:
    world = ros_session.fetch_world()
    WorldSynchronizer(_world=world, node=ros_session.node)
    ModelReloadSynchronizer(_world=world, node=ros_session.node)

robot_view = world.get_semantic_annotations_by_type(DAiSy)[0]


def perform(plan: StatechartNode, with_collision_avoidance: bool) -> None:
    """
    Perform `plan` with a new executor of :data:`executor_type`.

    :param plan: The plan to perform.
    :param with_collision_avoidance: Whether DAiSy avoids collisions while performing
        it.
    """
    executor = executor_type(
        context=StatechartContext.with_extensions(
            world=world, extensions=[RobotAccess(robot=robot_view)]
        ),
        ros_node=ros_session.node,
        collision_avoidance=with_collision_avoidance,
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=plan)
    try:
        executor.compile(statechart=statechart)
        executor.execute()
    finally:
        executor.close()


# %% Define Additional Objects


def merge_mesh(
    file_name: str, parent: Body, parent_T_mesh: HomogeneousTransformationMatrix
) -> Body:
    """
    :param file_name: The STL file in :data:`objects_directory`, which also names the
        body.
    :param parent: The body the mesh is fixed to.
    :param parent_T_mesh: Where the mesh is fixed relative to `parent`.
    :return: The body of the mesh, merged into the world unless it is there already.
    """
    existing = world.get_bodies_by_name(PrefixedName(file_name))
    if existing:
        return existing[0]
    mesh_world = STLParser(file_path=str(objects_directory / file_name)).create_world()
    mesh_root = mesh_world.root
    with world.modify_world():
        world.merge_world(
            mesh_world,
            FixedConnection(
                parent=parent,
                child=mesh_root,
                parent_T_connection_expression=parent_T_mesh,
            ),
        )
    return mesh_root


cable_post = merge_mesh(
    "item_profile_8_40x40_720.stl",
    robot_view.root,
    HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.625,
        y=0.42,
        z=0.8477,
        roll=pi / 2,
        yaw=-pi / 2,
        reference_frame=robot_view.root,
    ),
)

hanger_body = merge_mesh(
    "cable_hanger_2.stl",
    cable_post,
    HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.0,
        y=0.310,  # 720/2 - 50
        z=0.02,
        roll=-pi / 2,
        reference_frame=cable_post,
    ),
)

# %% Cable Definition
# TODO: Figure out how to respawn the cable at a new position
with world.modify_world():
    cable_annotation = Cable.create_with_new_body_in_world(
        name=PrefixedName("cable"),
        world=world,
        hanging_from=hanger_body,
        length=0.3,
        mount_offset_x=0.055,
        mount_offset_y=-0.07,
        height_offset=0.0,
    )
    world.collision_manager.add_ignore_collision_rule(
        AllowCollisionForBodies(allowed_collision_bodies={cable_annotation.root})
    )

# %% Debug Prints
if verbose:
    print(world.root.name)

    # Print joint states
    for dof in robot_view.degrees_of_freedom_with_hardware_interface:
        print(f"{dof.name}: {dof.variables.position.resolve():.2f}")

# %% Home Robot

daisy_left_arm_names = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

daisy_safe_left_arm_positions = [
    -2.71,  # left_shoulder_pan_joint
    -1.01,  # left_shoulder_lift_joint
    -2.10,  # left_elbow_joint
    -1.59,  # left_wrist_1_joint
    1.53,  # left_wrist_2_joint
    -4.23,  # left_wrist_3_joint
]

daisy_right_arm_names = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]

daisy_safe_right_arm_positions = [
    -0.02,  # right_shoulder_pan_joint
    -0.97,  # right_shoulder_lift_joint
    -2.00,  # right_elbow_joint
    -1.76,  # right_wrist_1_joint
    1.56,  # right_wrist_2_joint
    -0.83,  # right_wrist_3_joint
]


def move_joints(names: list[str], positions: list[float]) -> JointPositionList:
    """
    :param names: The names of the joints to move.
    :param positions: The position each joint in `names` is moved to.
    :return: A new goal moving the joints to `positions`.
    """
    return JointPositionList(
        goal_state=JointState(
            connections=[world.get_connection_by_name(name) for name in names],
            target_values=positions,
        )
    )


left_gripper = robot_view.left_arm.end_effector
right_gripper = robot_view.right_arm.end_effector


def both_grippers(
    configuration_of: Callable[[EndEffector], GripperConfiguration],
) -> Sequence:
    """
    Move the left, then the right griplink gripper, one after the other since moving
    them in parallel fails on the real robot.

    :param configuration_of: The configuration to move a gripper into.
    """
    return Sequence(
        nodes=[
            SetGripperAction(configuration=configuration_of(left_gripper)),
            SetGripperAction(configuration=configuration_of(right_gripper)),
        ]
    )


plan_home = Sequence(
    nodes=[
        both_grippers(lambda gripper: gripper.default_configuration(GripperState.OPEN)),
        both_grippers(
            lambda gripper: gripper.default_configuration(GripperState.CLOSE)
        ),
        both_grippers(lambda gripper: gripper.default_configuration(GripperState.OPEN)),
        ParkArmsAction(arms=robot_view.all_arms),
        # TODO Move this motion to the CableGraspAction and add generic version depending on the arm
        move_joints(daisy_left_arm_names, daisy_safe_left_arm_positions),
        both_grippers(
            lambda gripper: GriplinkFlexConfiguration.from_state_type(
                gripper,
                GripperState.FLEXCLOSE,
                grip_position=0.07,
                grip_velocity=0.3,
            )
        ),
    ]
)

perform(plan_home, with_collision_avoidance=False)

sleep(3)

# %% Demo Plan
perform(
    CableGraspAction(
        cable_annotation=cable_annotation,
        hanger_body=hanger_body,
        grasp_offset=0.1,
        side_offset=0.2,
        front_offset=-0.04,
        down_offset=0.1677,
        approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
        approach_sign=-1,  # y-axis pointing to the back
    ),
    with_collision_avoidance=collision_avoidance,
)

# %% Regrasp Cable
perform(
    Sequence(
        nodes=[
            # TODO: this should be in the action description
            ParkArmsAction(arms=[robot_view.right_arm]),
            # TODO Move this motion to the CableGraspAction and add generic version depending on the arm
            move_joints(daisy_right_arm_names, daisy_safe_right_arm_positions),
        ]
    ),
    with_collision_avoidance=collision_avoidance,
)
perform(
    CableRegraspAction(
        cable_annotation=cable_annotation,
        hanger_body=hanger_body,
        approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
        approach_sign=-1,  # y-axis pointing to the back
        regrasp_height=0.6,  # height above the table to perform the regrasp
    ),
    with_collision_avoidance=collision_avoidance,
)

# %% Rehang Cable
perform(
    Sequence(
        nodes=[
            # TODO: this should be in the action description
            ParkArmsAction(arms=[robot_view.right_arm]),
            # TODO Move this motion to the CableGraspAction and add generic version depending on the arm
            move_joints(daisy_right_arm_names, daisy_safe_right_arm_positions),
        ]
    ),
    with_collision_avoidance=collision_avoidance,
)
perform(
    CableRehangAction(
        cable_annotation=cable_annotation,
        hanger_body=hanger_body,
        side_offset=0.05,
        front_offset=0.07,
        up_offset=0.03,
        approach_direction=1,  # approach in y direction, coming from the front of the cable hanger
        approach_sign=-1,  # y-axis pointing to the back
    ),
    with_collision_avoidance=collision_avoidance,
)

print("Plan finished.")
ros_session.stop()
