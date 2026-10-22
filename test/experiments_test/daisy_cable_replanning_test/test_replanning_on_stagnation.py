"""
Replanning the motions of the cable demo that collision avoidance leaves stuck.
"""

from __future__ import annotations

import pytest
from experiments.daisy_cable_replanning.collisions import ArmCollisionPairs
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.utils import daisy_installed
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")

# %% collision pairs


def test_arm_pairs_ignore_temporary_rules_allowing_the_arm_to_collide(
    parked_daisy_world: World,
):
    robot = parked_daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    pairs = ArmCollisionPairs(world=parked_daisy_world, robot=robot)
    checked_before = pairs.collision_matrix_of(robot.right_arm, 0.0).collision_checks
    parked_daisy_world.collision_manager.add_temporary_rule(
        AllowCollisionForBodies(
            allowed_collision_bodies=pairs.bodies_of_arm[robot.right_arm]
        )
    )

    checked_after = pairs.collision_matrix_of(robot.right_arm, 0.0).collision_checks

    assert checked_before != set()
    assert checked_after == checked_before
