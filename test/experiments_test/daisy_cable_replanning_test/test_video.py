"""
Putting the videos of a seed with and without replanning next to each other.
"""

from __future__ import annotations

import numpy as np
from experiments.daisy_cable_replanning.video import SideBySideVideo

# %% side by side


def _frames(count: int, value: int) -> list[np.ndarray]:
    return [np.full((40, 60, 3), value, dtype=np.uint8) for _ in range(count)]


def test_side_by_side_lasts_as_long_as_the_longer_video():
    video = SideBySideVideo(
        left=_frames(3, 10), right=_frames(5, 200), left_caption="", right_caption=""
    )

    frames = video.frames

    assert len(frames) == 5
    assert frames[0].shape == (40, 120, 3)


def test_shorter_video_holds_its_last_frame():
    left = _frames(2, 10)
    left[-1][:] = 90
    video = SideBySideVideo(
        left=left, right=_frames(4, 200), left_caption="", right_caption=""
    )

    assert np.all(video.frames[-1][:, :60] == 90)
    assert np.all(video.frames[-1][:, 60:] == 200)
