"""
The minimal cable demonstration, run headless in MuJoCo with the cable in SuperDex.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.robots.daisy import DAiSy

pytest.importorskip("superdex.physics")

from experiments.minimal_cable_demo.demo import MinimalCableDemonstration

pytestmark = pytest.mark.skipif(
    not DAiSy.is_description_installed(), reason="DAiSy not installed"
)


# %% handover


def test_the_left_gripper_takes_the_cable_over_and_drops_it_onto_the_table():
    """
    Once the right gripper has let go, the left gripper alone holds the cable up; a
    cable it failed to take would lie on the table, one radius above it.

    Once the left gripper has dropped it, the cable lies on the table, every node within
    a few radii of it.
    """
    demonstration = MinimalCableDemonstration(headless=True)

    outcome = demonstration.run()

    assert (
        outcome.lowest_node_height_above_table_after_handover
        > 10 * demonstration.cable_radius
    )
    assert (
        outcome.highest_node_height_above_table_after_drop
        < 3 * demonstration.cable_radius
    )
