"""
MongoDB storage reader interface for RoboKudo.

This module provides an interface for reading stored sensor data and annotations
from a MongoDB database. It supports:

* Reading stored RGB-D camera data
* Loading camera calibration information
* Restoring annotations and views
* Automatic cursor management
* Optional looping through stored data

The module is primarily used for:

* Replaying recorded data
* Testing and debugging pipelines
* Offline data analysis
* Visualization of stored data
"""
import robokudo.io.camera_interface
import robokudo.io.storage

import open3d as o3d

from robokudo.cas import CASViews
from robokudo.annotator_parameters import AnnotatorPredefinedParameters


class StorageReaderInterface(robokudo.io.camera_interface.CameraInterface):
    """
    A camera interface for reading data from MongoDB storage.

    This interface reads sensor data and annotations that were previously stored
    using the StorageWriter annotator. It handles data deserialization and
    restoration of the Common Analysis Structure (CAS) views.

    :ivar storage: MongoDB storage interface
    :type storage: robokudo.io.storage.Storage
    :ivar reader: List-based reader for MongoDB data
    :type reader: robokudo.io.storage.Storage.ListReader
    """

    def __init__(self, camera_config):
        """
        Initialize the storage reader interface.

        Sets up MongoDB connection and creates a list reader for the specified
        database.

        :param camera_config: Configuration containing database settings
        :type camera_config: Any
        """
        super().__init__(camera_config)
        self.storage = robokudo.io.storage.Storage(camera_config.db_name)
        self.reader = self.storage.ListReader(camera_config.db_name)

    def has_new_data(self):
        """
        Check if more data is available to read.

        Handles looping behavior based on camera configuration and
        maintains cursor position in the data sequence.

        :return: True if more data is available, False otherwise
        :rtype: bool
        """
        # Check if we have to reinitialize the cursor after we hit the end of the recorded data
        if self.camera_config.loop and not self.reader.cursor_has_frames():
            self.reader.reset_cursor()

        return self.reader.cursor_has_frames()

    def set_data(self, cas: robokudo.cas.CAS):
        """
        Update the Common Analysis Structure with data from storage.

        This method:
        * Retrieves the next frame from storage
        * Restores views and annotations
        * Updates camera intrinsics
        * Sets depth availability flag

        :param cas: Common Analysis Structure to update
        :type cas: robokudo.cas.CAS
        """
        cas_frame = self.reader.get_next_frame()
        # Restore the views from the individual documents
        cas_frame['views'] = {}
        self.storage.load_views_from_mongo_in_cas(cas_frame)

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in cas_frame['views'].items():
            cas.set(view_name, view_content)

        # Restore annotations
        self.storage.load_annotations_from_mongo_in_cas(cas_frame, cas)

        if cas.get(CASViews.DEPTH_IMAGE) is None:
            # no depth image available
            AnnotatorPredefinedParameters.global_with_depth = False

        # Compute the cam intrinsic from the cam info.
        # this is harder to serialize and put transparently into mongo as of today, so we'll do it here
        # Construct o3d camera intrinsics from cam info in CAS

        # TODO this can be unified with the cam intrinsic creation in the cam interface. Check
        #  ROSCamInterface.set_o3d_cam_intrinsics_from_ros_cam_info
        #  => type_conversion.o3d_cam_intrinsics_from_ros_cam_info(cam_info):
        if "cam_info" in cas_frame['views']:
            cam_info = cas_frame['views']["cam_info"]

            if cam_info is None:
                # nothing to do
                return

            width = cam_info.width
            height = 960  # we assume that the kinect right now only outputs 1280x... images with the 4:3 crop
            fx = cam_info.k[0]
            cx = cam_info.k[2]
            fy = cam_info.k[4]
            cy = cam_info.k[5]
            cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            cas.set(CASViews.CAM_INTRINSIC, cam_intrinsic)
