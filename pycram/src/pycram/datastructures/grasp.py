from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from semantic_digital_twin.robots.abstract_robot import Manipulator, AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Optional, Union, List
from scipy.spatial.transform import Rotation as R

from .dataclasses import Rotations
from .enums import Grasp, AxisIdentifier, ApproachDirection, VerticalAlignment
from .pose import PoseStamped, PyCramVector3
from ..has_parameters import HasParameters
from ..tf_transformations import quaternion_multiply
from ..utils import translate_pose_along_local_axis



@dataclass
class NewGraspDescription:
    approach_direction: ApproachDirection
    """
    The direction from which the body should be grasped. These are the four directions in the x-y plane (FRONT, BACK, LEFT, RIGHT).
    """
    vertical_alignment: VerticalAlignment

    """
    The alignment of the gripper with the body in the z-axis (TOP, BOTTOM).
    """

    body: Body
    """
    The body that should be grasped.
    """

    manipulator: Manipulator
    """
    The manipulator that is used to grasp the body.
    """

    manipulation_offset: float = 0.05
    """
    The offset between the center of the pose in the grasp sequence
    """

    @classmethod
    def calculate_all_grasp_descriptions(cls, body: Body, manipulator: Manipulator):
        pass

    def grasp_pose_sequence(self, reverse: bool = False):
        """
        Calculates the pose sequence to grasp the body. The sequence is 3 poses, one in front of the body (taking body
        geometry into account), one at the center of the body, and the last one above the body to lift it.

        :param reverse: Indicates if the sequence should be reversed.
        :return: The pose sequence.
        """

        grasp_orientation = self.grasp_orientation()

        bb_in_frame = self.body.collision.as_bounding_box_collection_in_frame(self.body).bounding_box()

        grasp_axis = np.array(self.manipulation_axis(), dtype=np.bool)

        approach_axis = np.array(self.approach_direction.axis.value, dtype=np.bool)

        # Pre-pose calculation

        pre_position  = np.array([0., 0., 0.])
        offset = np.array(bb_in_frame.dimensions)[approach_axis] + self.manipulation_offset
        pre_position[grasp_axis] = -offset

        pre_pose = PoseStamped.from_list(pre_position.tolist(), grasp_orientation, self.body)

        # Lift pose calculation

        lift_axis = np.array(self.lift_axis(), dtype=np.bool)

        lift_position = np.array([0., 0., 0.])
        lift_position[lift_axis] = self.manipulation_offset
        lift_pose = PoseStamped.from_list(lift_position.tolist(), grasp_orientation, self.body)

        sequence = [pre_pose, PoseStamped.from_list([0, 0, 0], grasp_orientation, frame=self.body), lift_pose]

        if reverse:
            sequence.reverse()
        return sequence

    def manipulation_axis(self) -> List[float]:
        """
        Axis of the manipulator that is manipulating the body. Translates the x-axis of the global frame to how the
        manipulator is rotated.

        :returns: The axis of the manipulator that is manipulating the body.
        """
        return self.calculate_manipulator_axis([1, 0, 0])


    def lift_axis(self) -> List[float]:
        """
        Axis of the manipulator that is lifting the body. Translates the z-axis of the global frame to how the
        manipulator is rotated.

        :returns: The axis of the manipulator that is lifting the body.
        """
        return self.calculate_manipulator_axis([0, 0, 1])


    def calculate_manipulator_axis(self, axis: List[int]) -> List[float]:
        """
        Calculates the corresponding axis of the manipulator for a given axis of the body.

        :param axis: The axis of the body as a list of [x, y, z] indices.
        :return: The corresponding axis of the manipulator as a list of [x, y, z] values.
        """
        front_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            *axis, reference_frame=self.body
        )

        grasp_pose = self.grasp_pose_new()
        world = self.body._world

        front_global = world.transform(front_pose, world.root)

        grasp_global = world.transform(grasp_pose.to_spatial_type(), world.root)

        t = grasp_global.inverse().to_np() @ front_global.to_np()

        return t[:3, 3].astype(int).tolist()

    def grasp_orientation(self):
        """
        The orientation of the grasp pose in the global frame??. Takes into account the approach direction and vertical
        alignment.
        """
        rotation = Rotations.SIDE_ROTATIONS[self.approach_direction]
        rotation = quaternion_multiply(
            rotation, Rotations.VERTICAL_ROTATIONS[self.vertical_alignment]
        )
        rotation = quaternion_multiply(
            rotation, Rotations.HORIZONTAL_ROTATIONS[False]
        )

        orientation = quaternion_multiply(rotation, self.manipulator.front_facing_orientation.to_np())

        norm = math.sqrt(sum(comp**2 for comp in orientation))
        orientation = [comp / norm for comp in orientation]

        return orientation

    def edge_offset(self) -> float:
        """
        The offset between the center of the body and its edge in the direction of the approach axis.
        :return: The edge offset.
        """
        rim_direction_index = self.approach_direction.value[0].value.index(1)

        rim_offset = (
                self.body.collision.as_bounding_box_collection_in_frame(self.body)
                .bounding_box()
                .dimensions[rim_direction_index]
                / 2
        )
        return rim_offset

    def grasp_pose_new(self, grasp_edge: bool = False) -> PoseStamped:
        """
        The pose for the given manipulator to grasp the body in the frame of the body.

        :param grasp_edge: Indicates if the pose should be for the edge of the body or the center.
        :return: The pose of the body in the body frame.
        """
        edge_offset = -self.edge_offset() if grasp_edge else 0
        orientation = self.grasp_orientation()
        grasp_pose = PoseStamped().from_list([edge_offset, 0, 0], orientation, frame=self.body)

        return grasp_pose


# @has_parameters
@dataclass
class GraspDescription(HasParameters):
    """
    Represents a grasp description with a side grasp, top face, and orientation alignment.
    """

    approach_direction: ApproachDirection
    """
    The primary approach direction. 
    """

    vertical_alignment: VerticalAlignment = VerticalAlignment.NoAlignment
    """
    The vertical alignment when grasping the pose
    """

    rotate_gripper: bool = False
    """
    Indicates if the gripper should be rotated by 90°. Must be a boolean.
    """

    def __hash__(self):
        return hash(
            (self.approach_direction, self.vertical_alignment, self.rotate_gripper)
        )

    def as_list(self) -> List[Union[Grasp, Optional[Grasp], bool]]:
        """
        :return: A list representation of the grasp description.
        """
        return [self.approach_direction, self.vertical_alignment, self.rotate_gripper]

    def get_grasp_pose(
        self, end_effector: Manipulator, body: Body, translate_rim_offset: bool = False
    ) -> PoseStamped:
        """
        Translates the grasp pose of the object using the desired grasp description and object knowledge.
        Leaves the orientation untouched.
        Returns the translated grasp pose.

        :param end_effector: The end effector that will be used to grasp the object.
        :param body: The body of the object to be grasped.
        :param translate_rim_offset: If True, the grasp pose will be translated along the rim offset.

        :return: The grasp pose of the object.
        """
        grasp_pose = PoseStamped().from_spatial_type(body.global_pose)

        approach_direction = self.approach_direction
        rim_direction_index = approach_direction.value[0].value.index(1)

        rim_offset = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .dimensions[rim_direction_index]
            / 2
        )

        grasp_pose.rotate_by_quaternion(
            self.calculate_grasp_orientation(
                end_effector.front_facing_orientation.to_np()
            )
        )
        if translate_rim_offset:
            grasp_pose = translate_pose_along_local_axis(
                grasp_pose, self.approach_direction.axis.value, -rim_offset
            )

        return grasp_pose

    def calculate_grasp_orientation(self, front_orientation: np.ndarray) -> List[float]:
        """
        Calculates the grasp orientation based on the approach axis and the grasp description.

        :param front_orientation: The front-facing orientation of the end effector as a numpy array.

        :return: The calculated orientation as a quaternion.
        """
        rotation = Rotations.SIDE_ROTATIONS[self.approach_direction]
        rotation = quaternion_multiply(
            rotation, Rotations.VERTICAL_ROTATIONS[self.vertical_alignment]
        )
        rotation = quaternion_multiply(
            rotation, Rotations.HORIZONTAL_ROTATIONS[self.rotate_gripper]
        )

        orientation = quaternion_multiply(rotation, front_orientation)

        norm = math.sqrt(sum(comp**2 for comp in orientation))
        orientation = [comp / norm for comp in orientation]

        return orientation

    @staticmethod
    def calculate_grasp_descriptions(
        robot: AbstractRobot,
        pose: PoseStamped,
        grasp_alignment: Optional[PreferredGraspAlignment] = None,
    ) -> List[GraspDescription]:
        """
        This method determines the possible grasp configurations (approach axis and vertical alignment) of the body,
        taking into account the bodies orientation, position, and whether the gripper should be rotated by 90°.

        :param robot: The robot for which the grasp configurations are being calculated.
        :param grasp_alignment: An optional PreferredGraspAlignment object that specifies preferred grasp axis,
        :param pose: The pose of the object to be grasped.

        :return: A sorted list of GraspDescription instances representing all grasp permutations.
        """
        objectTmap = pose

        robot_pose = PoseStamped.from_spatial_type(robot.root.global_pose)

        if grasp_alignment:
            side_axis = grasp_alignment.preferred_axis
            vertical = grasp_alignment.with_vertical_alignment
            rotated_gripper = grasp_alignment.with_rotated_gripper
        else:
            side_axis, vertical, rotated_gripper = (
                AxisIdentifier.Undefined,
                False,
                False,
            )

        object_to_robot_vector_world = objectTmap.position.vector_to_position(
            robot_pose.position
        )
        orientation = objectTmap.orientation.to_list()

        mapRobject = R.from_quat(orientation).as_matrix()
        objectRmap = mapRobject.T

        object_to_robot_vector_local = objectRmap.dot(
            object_to_robot_vector_world.to_numpy()
        )
        vector_x, vector_y, vector_z = object_to_robot_vector_local

        vector_side = PyCramVector3(vector_x, vector_y, np.nan)
        side_faces = GraspDescription.calculate_closest_faces(vector_side, side_axis)

        vector_vertical = PyCramVector3(np.nan, np.nan, vector_z)
        if vertical:
            vertical_faces = GraspDescription.calculate_closest_faces(vector_vertical)
        else:
            vertical_faces = [VerticalAlignment.NoAlignment]

        grasp_configs = [
            GraspDescription(
                approach_direction=side,
                vertical_alignment=top_face,
                rotate_gripper=rotated_gripper,
            )
            for top_face in vertical_faces
            for side in side_faces
        ]

        return grasp_configs

    @staticmethod
    def calculate_closest_faces(
        pose_to_robot_vector: PyCramVector3,
        specified_grasp_axis: AxisIdentifier = AxisIdentifier.Undefined,
    ) -> Union[
        Tuple[ApproachDirection, ApproachDirection],
        Tuple[VerticalAlignment, VerticalAlignment],
    ]:
        """
        Determines the faces of the object based on the input vector.

        If `specified_grasp_axis` is None, it calculates the primary and secondary faces based on the vector's magnitude
        determining which sides of the object are most aligned with the robot. This will either be the x, y plane for side faces
        or the z axis for top/bottom faces.
        If `specified_grasp_axis` is provided, it only considers the specified axis and calculates the faces aligned
        with that axis.

        :param pose_to_robot_vector: A 3D vector representing one of the robot's axes in the pose's frame, with
                              irrelevant components set to np.nan.
        :param specified_grasp_axis: Specifies a specific axis (e.g., X, Y, Z) to focus on.

        :return: A tuple of two Grasp enums representing the primary and secondary faces.
        """
        all_axes = [AxisIdentifier.X, AxisIdentifier.Y, AxisIdentifier.Z]

        if not specified_grasp_axis == AxisIdentifier.Undefined:
            valid_axes = [specified_grasp_axis]
        else:
            valid_axes = [
                axis
                for axis in all_axes
                if not np.isnan(pose_to_robot_vector.to_list()[axis.value.index(1)])
            ]

        object_to_robot_vector = np.array(pose_to_robot_vector.to_list()) + 1e-9
        sorted_axes = sorted(
            valid_axes,
            key=lambda axis: abs(object_to_robot_vector[axis.value.index(1)]),
            reverse=True,
        )

        primary_axis: AxisIdentifier = sorted_axes[0]
        primary_sign = int(np.sign(object_to_robot_vector[primary_axis.value.index(1)]))

        primary_axis_class = (
            VerticalAlignment if primary_axis == AxisIdentifier.Z else ApproachDirection
        )
        primary_face = primary_axis_class.from_axis_direction(
            primary_axis, primary_sign
        )

        if len(sorted_axes) > 1:
            secondary_axis: AxisIdentifier = sorted_axes[1]
            secondary_sign = int(
                np.sign(object_to_robot_vector[secondary_axis.value.index(1)])
            )
        else:
            secondary_axis: AxisIdentifier = primary_axis
            secondary_sign = -primary_sign

        secondary_axis_class = (
            VerticalAlignment
            if secondary_axis == AxisIdentifier.Z
            else ApproachDirection
        )
        secondary_face = secondary_axis_class.from_axis_direction(
            secondary_axis, secondary_sign
        )

        return primary_face, secondary_face


@dataclass
class PreferredGraspAlignment:
    """
    Description of the preferred grasp alignment for an object.
    """

    preferred_axis: Optional[AxisIdentifier]
    """
    The preferred axis, X, Y, or Z, for grasping the object, or None if not specified.
    """

    with_vertical_alignment: bool
    """
    Indicates if the object should be grasped with a vertical alignment.
    """

    with_rotated_gripper: bool
    """
    Indicates if the gripper should be rotated by 90° around X.
    """
