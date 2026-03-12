import logging
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import rustworkx as rx
from box import Box
from scipy.spatial import ConvexHull
from sortedcontainers import SortedSet
from typing_extensions import List, Union, Iterable, Optional, Iterator, Tuple

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.qp_controller_config import QPControllerConfig
from probabilistic_model.distributions import (
    DiracDeltaDistribution,
    GaussianDistribution,
)
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_event,
    leaf,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    ProbabilisticCircuit,
    ProductUnit,
)
from pycram.datastructures.dataclasses import Context
from pycram.plans.designator import Designator
from random_events.interval import closed
from random_events.polytope import Polytope, NoOptimalSolutionError
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body
from pycram.config.action_conf import ActionConfig
from pycram.locations.costmaps import (
    OccupancyCostmap,
    VisibilityCostmap,
    GaussianCostmap,
    Costmap,
    OrientationGenerator,
    RingCostmap,
)
from pycram.datastructures.enums import (
    Arms,
    Grasp,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped, GraspPose, PyCramVector3

from pycram.failures import RobotInCollision
from pycram.pose_validator import (
    visibility_validator,
    collision_check,
    pose_sequence_reachability_validator,
)
from pycram.utils import link_pose_for_joint_config
from pycram.view_manager import ViewManager

logger = logging.getLogger("pycram")


@dataclass
class CostmapLocation:
    """
    Uses Costmaps to create locations for complex constrains
    """

    target: PoseStamped
    """
    The target pose for which the location should be calculated.
    """

    reachable: bool = False
    """
    Rather the costmap should be used for reaching or not.
    """

    visible: bool = False
    """
    Rather the costmap should be used for visibility or not.
    """

    reachable_arm: Optional[Arms] = None

    ignore_collision_with: Optional[Body] = None
    grasp_description: Optional[GraspDescription] = None

    context: Optional[Context] = None

    @property
    def world(self):
        return self.context.world

    @property
    def robot(self):
        return self.context.robot

    def setup_costmaps(
        self, target: PoseStamped, visible: bool, reachable: bool
    ) -> Costmap:
        """
        Sets up the costmap for the given target and robot.
        The costmap are merged and stored in the final_map
        """
        ground_pose = deepcopy(target)
        ground_pose.position.z = 0

        base_bb = self.robot.base.bounding_box

        occupancy = OccupancyCostmap(
            distance_to_obstacle=(base_bb.depth / 2 + base_bb.width / 2) / 2,
            world=self.world,
            robot_view=self.robot,
            width=200,
            height=200,
            resolution=0.02,
            origin=ground_pose,
        )
        final_map = occupancy

        if visible:
            camera = list(self.robot.neck.sensors)[0]
            visible = VisibilityCostmap(
                min_height=camera.minimal_height,
                max_height=camera.maximal_height,
                world=self.world,
                width=200,
                height=200,
                resolution=0.02,
                origin=target,
            )
            final_map += visible

        if reachable:
            ring = RingCostmap(
                resolution=0.02,
                width=200,
                height=200,
                std=15,
                distance=0.4,  # That needs to be replaced with an estimate of the reachability space of the robot arms
                world=self.world,
                origin=target,
            )
            final_map += ring

        return final_map

    def __iter__(self) -> Iterator[PoseStamped]:
        """
        Generates positions for a given set of constrains from a costmap and returns
        them. The generation is based of a costmap which itself is the product of
        merging costmaps, each for a different purpose. In any case an occupancy costmap
        is used as the base, then according to the given constrains a visibility or
        gaussian costmap is also merged with this. Once the costmaps are merged,
        a generator generates pose candidates from the costmap. Each pose candidate
        is then validated against the constraints given by the designator if all validators
        pass the pose is considered valid and yielded.

           :yield: An instance of CostmapLocation.Location with a valid position that satisfies the given constraints
        """

        test_world = deepcopy(self.world)
        test_world.name = "Test World"

        robot = self.robot

        test_robot = robot.from_world(test_world)

        objects_in_hand = list(
            set(test_world.get_kinematic_structure_entities_of_branch(test_robot.root))
            - set(test_robot.bodies)
        )
        object_in_hand = objects_in_hand[0] if objects_in_hand else None

        final_map = self.setup_costmaps(self.target, self.visible, self.reachable)
        final_map.number_of_samples = 600
        final_map.orientation_generator = (
            OrientationGenerator.orientation_generator_for_axis(
                list(self.robot.base.main_axis.to_np())
            )
        )

        for pose_candidate in final_map:
            logger.debug(f"Testing candidate pose at {pose_candidate}")
            pose_candidate.position.z = 0
            test_robot.root.parent_connection.origin = pose_candidate.to_spatial_type()

            collisions = collision_check(
                robot=test_robot,
                world=test_world,
            )

            if collisions:
                logger.debug(f"Candidate pose in collision, skipping")
                continue

            if not (self.reachable or self.visible):
                self._last_result = pose_candidate
                yield pose_candidate
                continue

            if self.visible and not visibility_validator(
                test_robot, self.target, test_world
            ):
                logger.debug(f"Candidate pose not visible, skipping")
                continue

            if not self.reachable:
                self._last_result = pose_candidate
                yield pose_candidate
                continue

            grasp_descriptions = (
                [self.grasp_description]
                if self.grasp_description
                else GraspDescription.calculate_grasp_descriptions(
                    ViewManager.get_arm_view(
                        self.reachable_arm, test_robot
                    ).manipulator,
                    self.target,
                )
            )

            for grasp_description in grasp_descriptions:
                target_sequence = grasp_description._pose_sequence(
                    self.target, object_in_hand
                )

                ee = ViewManager.get_arm_view(self.reachable_arm, test_robot)
                is_reachable = pose_sequence_reachability_validator(
                    target_sequence,
                    ee.manipulator.tool_frame,
                    test_robot,
                    test_world,
                    use_fullbody_ik=test_robot.full_body_controlled,
                )
                if is_reachable:
                    pose = GraspPose(
                        pose_candidate.pose,
                        pose_candidate.header,
                        arm=self.reachable_arm,
                        grasp_description=grasp_description,
                    )
                    self._last_result = pose
                    yield pose


class AccessingLocation:
    """
    Location designator which describes poses used for opening drawers
    """

    def __init__(
        self,
        handle: Union[Body, Iterable[Body]],
        robot_desig: Union[AbstractRobot, Iterable[AbstractRobot]],
        arm: Union[List[Arms], Arms] = None,
        prepose_distance: float = ActionConfig.grasping_prepose_distance,
    ):
        """
        Describes a position from where a drawer can be opened. For now this position should be calculated before the
        drawer will be opened. Calculating the pose while the drawer is open could lead to problems.

        :param handle: ObjectPart designator for handle of the drawer
        :param robot_desig: Object designator for the robot which should open the drawer
        """

        self.handle: Body = handle
        self.robot: AbstractRobot = robot_desig
        self.prepose_distance = prepose_distance
        self.arm = arm if arm is not None else [Arms.LEFT, Arms.RIGHT]

    def ground(self) -> PoseStamped:
        """
        Default specialized_designators for this location designator, just returns the first element from the iteration

        :return: A location designator for a pose from which the drawer can be opened
        """
        return next(iter(self))

    @staticmethod
    def adjust_map_for_drawer_opening(
        cost_map: Costmap,
        init_pose: PoseStamped,
        goal_pose: PoseStamped,
        width: float = 0.2,
    ):
        """
        Adjust the cost map for opening a drawer. This is done by removing all locations between the initial and final
        pose of the drawer/container.

        :param cost_map: Costmap that should be adjusted.
        :param init_pose: Pose of the drawer/container when it is fully closed.
        :param goal_pose: Pose of the drawer/container when it is fully opened.
        :param width: Width of the drawer/container.
        """
        motion_vector = [
            goal_pose.position.x - init_pose.position.x,
            goal_pose.position.y - init_pose.position.y,
            goal_pose.position.z - init_pose.position.z,
        ]
        # remove locations between the initial and final pose
        motion_vector_length = np.linalg.norm(motion_vector)
        unit_motion_vector = np.array(motion_vector) / motion_vector_length
        orthogonal_vector = np.array([unit_motion_vector[1], -unit_motion_vector[0], 0])
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)
        orthogonal_size = width
        map_origin_idx = cost_map.map.shape[0] // 2, cost_map.map.shape[1] // 2
        for i in range(int(motion_vector_length / cost_map.resolution)):
            for j in range(int(orthogonal_size / cost_map.resolution)):
                idx = (
                    int(
                        map_origin_idx[0]
                        + i * unit_motion_vector[0]
                        + j * orthogonal_vector[0]
                    ),
                    int(
                        map_origin_idx[1]
                        + i * unit_motion_vector[1]
                        + j * orthogonal_vector[1]
                    ),
                )
                cost_map.map[idx] = 0
                idx = (
                    int(
                        map_origin_idx[0]
                        + i * unit_motion_vector[0]
                        - j * orthogonal_vector[0]
                    ),
                    int(
                        map_origin_idx[1]
                        + i * unit_motion_vector[1]
                        - j * orthogonal_vector[1]
                    ),
                )
                cost_map.map[idx] = 0

    def setup_costmaps(self, handle: Body) -> Costmap:
        """
        Sets up the costmaps for the given handle and robot. The costmaps are merged and stored in the final_map.
        """
        ground_pose = PoseStamped.from_spatial_type(handle.global_pose)
        ground_pose.position.z = 0

        base_bb = self.robot_view.base.bounding_box
        occupancy = OccupancyCostmap(
            robot_view=self.robot_view,
            distance_to_obstacle=(base_bb.depth / 2 + base_bb.width / 2) / 2,
            width=200,
            height=200,
            resolution=0.02,
            origin=ground_pose,
            world=handle._world,
        )
        final_map = occupancy

        gaussian = GaussianCostmap(200, 15, handle._world, 0.02, ground_pose)
        final_map += gaussian

        return final_map

    def create_target_sequence(
        self, params_box: Box, final_map: Costmap
    ) -> List[PoseStamped]:
        """
        Creates the sequence of target poses

        :param params_box:
        :param final_map:
        :return:
        """
        handle = params_box.handle
        # compute the chain of connections only works top down,
        handle_to_root_connections = list(
            reversed(
                handle._world.compute_chain_of_connections(handle._world.root, handle)
            )
        )
        # Search for the first connection that is not a FixedConnection,
        container_connection = list(
            filter(
                lambda c: not isinstance(c, FixedConnection), handle_to_root_connections
            )
        )[0]

        lower_limit = container_connection.dof.limits.lower.position
        upper_limit = container_connection.dof.limits.upper.position

        init_pose = link_pose_for_joint_config(
            params_box.handle, {container_connection.dof.name.name: lower_limit}
        )

        # Calculate the pose the handle would be in if the drawer was to be fully opened
        goal_pose = link_pose_for_joint_config(
            params_box.handle, {container_connection.dof.name.name: upper_limit}
        )

        # Handle position for calculating rotation of the final pose
        half_pose = link_pose_for_joint_config(
            params_box.handle, {container_connection.dof.name.name: upper_limit / 1.5}
        )

        # joint_type = params_box.handle.parent_entity.joints[container_joint].type

        # if joint_type == JointType.PRISMATIC:
        #     self.adjust_map_for_drawer_opening(final_map, init_pose, goal_pose)

        target_sequence = [init_pose, half_pose, goal_pose]
        return target_sequence

    def __iter__(self) -> Iterator[PoseStamped]:
        """
        Creates poses from which the robot can open the drawer specified by the ObjectPart designator describing the
        handle. Poses are validated by checking if the robot can grasp the handle while the drawer is closed and if
        the handle can be grasped if the drawer is open.

        :yield: A location designator containing the pose and the arms that can be used.
        """
        test_world = deepcopy(self.world)
        test_robot = self.robot_view.from_world(test_world)
        for params in self.generate_permutations():
            params_box = Box(params)

            final_map = self.setup_costmaps(params_box.handle)

            target_sequence = self.create_target_sequence(params_box, final_map)
            half_pose = target_sequence[1]

            orientation_generator = (
                lambda p, o: OrientationGenerator.generate_origin_orientation(
                    p, half_pose
                )
            )
            final_map.number_of_samples = 600
            final_map.orientation_generator = orientation_generator
            for pose_candidate in final_map:
                pose_candidate.position.z = 0
                test_robot.root.parent_connection.origin = (
                    pose_candidate.to_spatial_type()
                )
                try:
                    collision_check(test_robot, test_world)
                except RobotInCollision:
                    continue

                for arm_chain in test_robot.manipulator_chains:
                    grasp = GraspDescription(
                        ApproachDirection.FRONT,
                        VerticalAlignment.NoAlignment,
                        arm_chain.manipulator,
                    ).grasp_orientation()
                    current_target_sequence = [
                        deepcopy(pose) for pose in target_sequence
                    ]
                    for pose in current_target_sequence:
                        pose.rotate_by_quaternion(grasp)

                    is_reachable = pose_sequence_reachability_validator(
                        current_target_sequence,
                        arm_chain.manipulator.tool_frame,
                        test_robot,
                        test_world,
                        use_fullbody_ik=test_robot.full_body_controlled,
                    )
                    if is_reachable:
                        yield pose_candidate
