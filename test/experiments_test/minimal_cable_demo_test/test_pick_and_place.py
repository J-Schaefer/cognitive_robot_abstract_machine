"""
The cable pick-and-place demonstration, run headless in MuJoCo with the cable in
SuperDex.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from semantic_digital_twin.robots.daisy import DAiSy

pytest.importorskip("superdex.physics")

from experiments.minimal_cable_demo.daisy_cable_scene import CableVideo
from experiments.minimal_cable_demo.pick_and_place import (
    CablePickAndPlaceDemonstration,
)

pytestmark = pytest.mark.skipif(
    not DAiSy.is_description_installed(), reason="DAiSy not installed"
)

# %% picking and placing


def test_the_cable_lies_flat_on_the_table_near_the_place_point_after_it_was_laid_down():
    """
    Once the gripper has let go, the cable lies on the table again, every node within a
    few radii of it, with the point the gripper held close to where it was to be laid.

    A cable the gripper dropped on the way would lie where it fell.
    """
    demonstration = CablePickAndPlaceDemonstration(headless=True)

    outcome = demonstration.run()

    assert outcome.highest_node_height_above_table < 3 * demonstration.cable_radius
    assert outcome.distance_from_place_point < 0.05


# %% video


def test_a_video_holds_one_frame_per_frame_period_of_simulated_time(tmp_path):
    """
    Stepping the simulation captures a frame each time a frame period of simulated time
    has passed, and the frames are written into the video's file.
    """
    video = CableVideo(path=tmp_path / "cable.mp4", width=160, height=120)
    demonstration = CablePickAndPlaceDemonstration(headless=True, video=video)
    world, _ = demonstration.create_world()
    demonstration.lay_cable_on_the_table(world=world)
    simulation = demonstration.create_simulation(world=world)
    frames = 6
    try:
        demonstration.let_the_cable_settle(
            simulation=simulation,
            duration=frames * timedelta(seconds=1 / video.frames_per_second),
        )
    finally:
        simulation.stop_simulation()
    video.write()

    assert len(video.frames) == frames
    assert video.frames[0].shape == (video.height, video.width, 3)
    assert video.path.stat().st_size > 0
