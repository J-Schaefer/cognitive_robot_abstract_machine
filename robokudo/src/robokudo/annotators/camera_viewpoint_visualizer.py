"""Camera viewpoint visualization for RoboKudo.

This module provides an annotator for visualizing camera viewpoints and reference frames
in 3D space using Open3D.
"""

import numpy as np
import open3d as o3d

from py_trees.common import Status
from . import core
from ..types.tf import StampedTransform
from ..utils.transform import get_transform_matrix_from_q
from ..cas import CASViews


class CameraViewpointVisualizer(core.BaseAnnotator):
    """Annotator for visualizing camera viewpoints and reference frames.

    This annotator displays the reference frame set in the viewpoint (e.g., /map frame)
    along with the point cloud data. It creates a 3D visualization showing the
    coordinate frame and the point cloud in the same space.

    The annotator will fail if the required viewpoint transform cannot be found in the CAS.
    """

    def __init__(self, name: str = "CameraViewpointVisualizer"):
        """Initialize the camera viewpoint visualizer.

        :param name: Name of the annotator instance, defaults to "CameraViewpointVisualizer"
        """
        super().__init__(name)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """Update the visualization with the current viewpoint and point cloud data.

        Creates a visualization containing:

        * A coordinate frame representing the world reference frame
        * The current point cloud data

        :return: SUCCESS if visualization was created, FAILURE if viewpoint not found
        :raises AssertionError: If viewpoint transform is of a wrong type
        """
        try:
            cam_to_world_transform = self.get_cas().get(CASViews.VIEWPOINT_CAM_TO_WORLD)
        except Exception as err:
            self.rk_logger.warning(f"Couldn't find viewpoint in the CAS: {err}")
            return Status.FAILURE

        assert isinstance(cam_to_world_transform, StampedTransform)

        cloud = self.get_cas().get(CASViews.CLOUD)
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        t = get_transform_matrix_from_q(
            cam_to_world_transform.rotation, cam_to_world_transform.translation
        )
        t = np.linalg.inv(t)
        world_frame.transform(t)

        geometries = [
            {"name": "World frame", "geometry": world_frame},
            {"name": "Cloud", "geometry": cloud},
        ]
        self.get_annotator_output_struct().set_geometries(geometries)

        return Status.SUCCESS
