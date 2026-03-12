from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from typing_extensions import Optional, Any

from pycram.config.action_conf import ActionConfig
from pycram.datastructures.pose import PoseStamped
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from pycram.tf_transformations import quaternion_from_euler


@dataclass
class FaceAtAction(ActionDescription):
    """
    Turn the robot chassis such that is faces the ``pose`` and after that perform a look at action.
    """

    pose: PoseStamped
    """
    The pose to face 
    """
    keep_joint_states: bool = ActionConfig.face_at_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def execute(self) -> None:
        # get the robot position
        robot_position = PoseStamped.from_spatial_type(self.robot.root.global_pose)

        # calculate orientation for robot to face the object
        angle = (
            np.arctan2(
                robot_position.position.y - self.pose.position.y,
                robot_position.position.x - self.pose.position.x,
            )
            + np.pi
        )
        orientation = list(quaternion_from_euler(0, 0, angle, axes="sxyz"))

        # create new robot pose
        new_robot_pose = PoseStamped.from_list(
            robot_position.position.to_list(), orientation, self.world.root
        )

        self.add_subplan(
            sequential(
                [
                    NavigateAction(
                        new_robot_pose, self.keep_joint_states
                    ),  # turn robot
                    LookAtAction(self.pose),  # look at the target
                ]
            )
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        # The validation will be done in the LookAtActionPerformable.perform() method so no need to validate here.
        pass
