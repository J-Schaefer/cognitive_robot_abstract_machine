"""
The minimal cable demonstration, run headless in MuJoCo with the cable in SuperDex.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.utils import daisy_installed

pytest.importorskip("superdex.physics")

from experiments.minimal_cable_demo.demo import MinimalCableDemonstration

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")


def test_the_left_gripper_keeps_the_cable_off_the_table_after_the_handover():
    """
    Once the right gripper has let go, the left gripper alone holds the cable up; a
    dropped cable would lie on the table, one radius above it.
    """
    demonstration = MinimalCableDemonstration(headless=True)

    outcome = demonstration.run()

    assert outcome.lowest_node_height_above_table > 10 * demonstration.cable_radius
