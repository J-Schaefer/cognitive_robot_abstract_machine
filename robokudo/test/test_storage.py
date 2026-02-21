import copy
import os

import open3d as o3d
import pytest
import pymongo
import numpy as np

import robokudo.cas
from robokudo.cas import CAS, CASViews
from robokudo.io.storage import Storage
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis


@pytest.fixture(scope="module")
def storage_instance():
    db_name = "ONLY_UNITTESTS_test_db"
    storage = Storage(db_name)
    yield storage
    # Cleanup after tests
    storage.drop_database()


@pytest.fixture()
def cas_data():
    cas = CAS()
    """
    Mock CAS data for testing.
    """
    cas.set(CASViews.COLOR_IMAGE, np.random.rand(100, 100, 3))
    cas.set(CASViews.DEPTH_IMAGE, np.random.rand(100, 100))

    return cas


def store_cas_in_storage(storage_instance, cas_data):
    flat_cas = storage_instance.generate_dict_from_real_cas(cas_data)
    flat_cas['view_ids'] = {}

    # step 1: persist each view
    storage_instance.store_views_in_mongo(flat_cas)

    # step 2: persist cas and pointers to each view
    del flat_cas['views']
    return storage_instance.store_cas_dict(flat_cas)


def create_point_cloud():
    """
    Create an Open3D point cloud with some sample data.
    """
    # Create a numpy array of points
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Create the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud


class TestStorage:
    def test_drop_database(self, storage_instance):
        db_name = storage_instance.db_name
        storage_instance.drop_database()

        # Verify the database no longer exists

        # Fetch environment variables which might have been used to configure
        # another MongoDB Host and Port
        # This was originally introduced to support unit tests.
        mongo_host = os.getenv("RK_MONGO_HOST", "localhost")
        mongo_port = int(os.getenv("RK_MONGO_PORT", 27017))
        client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
        assert db_name not in client.list_database_names()

    def test_store_and_retrieve_cas(self, storage_instance, cas_data):
        result = store_cas_in_storage(storage_instance, cas_data)
        # Generate a dict from the CAS and store it
        flat_cas = storage_instance.generate_dict_from_real_cas(cas_data)
        flat_cas['view_ids'] = {}

        # step 1: persist each view
        storage_instance.store_views_in_mongo(flat_cas)

        # step 2: persist cas and pointers to each view
        del flat_cas['views']
        result = storage_instance.store_cas_dict(flat_cas)

        assert result.acknowledged

        # Retrieve the stored CAS
        retrieved_cas = storage_instance.db.cas.find_one({'_id': result.inserted_id})
        assert retrieved_cas is not None
        assert retrieved_cas['timestamp'] == cas_data.timestamp
        assert retrieved_cas['timestamp_readable'] == cas_data.timestamp_readable

    def test_nd_array_to_numpy_binary(self):
        array = np.random.rand(10, 10)
        binary = Storage.nd_array_to_numpy_binary(array)
        restored_array = Storage.numpy_binary_to_nd_array(binary)

        assert np.array_equal(array, restored_array)

    def test_store_and_retrieve_cas_with_single_oh(self, storage_instance, cas_data):
        # Generate a dict from the CAS and store it
        oh = ObjectHypothesis()
        oh.source = 'test_storage'
        oh.id = 1
        oh.roi = ImageROI()
        oh.roi.roi.pos.x = 1
        oh.roi.roi.pos.y = 2
        oh.roi.roi.width = 20
        oh.roi.roi.height = 40
        oh.points = create_point_cloud()

        cas_data.annotations.append(oh)

        result = store_cas_in_storage(storage_instance, cas_data)

        flat_cas = storage_instance.generate_dict_from_real_cas(cas_data)
        flat_cas['view_ids'] = {}

        # step 1: persist each view
        storage_instance.store_views_in_mongo(flat_cas)

        # step 2: persist cas and pointers to each view
        del flat_cas['views']
        result = storage_instance.store_cas_dict(flat_cas)

        assert result.acknowledged

        # Retrieve the stored CAS
        retrieved_cas_record = storage_instance.db.cas.find_one({'_id': result.inserted_id})
        assert retrieved_cas_record is not None
        assert retrieved_cas_record['timestamp'] == cas_data.timestamp
        assert retrieved_cas_record['timestamp_readable'] == cas_data.timestamp_readable

        retrieved_cas = robokudo.cas.CAS()
        retrieved_cas_record['views'] = {}
        storage_instance.load_views_from_mongo_in_cas(retrieved_cas_record)

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in retrieved_cas_record['views'].items():
            retrieved_cas.set(view_name, view_content)

        storage_instance.load_annotations_from_mongo_in_cas(retrieved_cas_record, retrieved_cas)

        # Compare the most important views: COLOR and DEPTH Image
        assert np.array_equal(cas_data.get(CASViews.COLOR_IMAGE), retrieved_cas.get(CASViews.COLOR_IMAGE))
        assert np.array_equal(cas_data.get(CASViews.DEPTH_IMAGE), retrieved_cas.get(CASViews.DEPTH_IMAGE))

        assert len(retrieved_cas.annotations) == 1
        retrieved_oh = retrieved_cas.annotations[0]

        assert oh.source == retrieved_oh.source
        assert oh.id == retrieved_oh.id
        assert oh.roi.roi.pos.x == retrieved_oh.roi.roi.pos.x
        assert oh.roi.roi.pos.y == retrieved_oh.roi.roi.pos.y
        assert oh.roi.roi.width == retrieved_oh.roi.roi.width
        assert oh.roi.roi.height == retrieved_oh.roi.roi.height

        # Compare PointCloud
        test_pcd = create_point_cloud()

        np.testing.assert_array_equal(
            np.asarray(test_pcd.points),
            np.asarray(retrieved_oh.points.points)
        )
