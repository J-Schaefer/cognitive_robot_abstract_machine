"""
Fixtures of the DAiSy cable replanning experiment: a DAiSy world of its own per test,
since the tests attach callbacks to it and move its arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.utils import daisy_installed
from semantic_digital_twin.world import World

# %% DAiSy world


@pytest.fixture()
def parked_daisy_world() -> World:
    """
    A DAiSy world with both arms parked.
    """
    if not daisy_installed():
        pytest.skip("DAiSy not installed")
    world = URDFParser.from_file(file_path=DAiSy.get_ros_file_path()).parse()
    DAiSy.from_world(world)
    robot = world.get_semantic_annotations_by_type(DAiSy)[0]
    for arm in robot.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    return world


@pytest.fixture()
def right_forearm_touching_gripper() -> np.ndarray:
    """
    Positions of both DAiSy arms, left arm first, in which the right forearm reaches
    into the side cylinder of the right gripper.
    """
    data = Path(__file__).parent / "right_forearm_touches_gripper.json"
    return np.array(json.loads(data.read_text())["positions"])
