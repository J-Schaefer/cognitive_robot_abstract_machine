from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from experiments.daisy_cable_replanning.summary import ExperimentOutcome
from experiments.daisy_cable_replanning.trials import CableDemoTrial, TrialFiles
from PIL import Image, ImageDraw, ImageFont

# %% side by side


class VideoCaption(StrEnum):
    """
    The captions of the two halves of the comparison video.
    """

    WITHOUT_REPLANNING = "Giskard only"
    WITH_REPLANNING = "Giskard + retract + OMPL replanning"
    WITH_COLLISION_AVOIDANCE = "Giskard with collision avoidance"
    WITH_STAGNATION_REPLANNING = "Giskard with collision avoidance + OMPL when stuck"


@dataclass
class SideBySideVideo:
    """
    Two videos played next to each other, each with a caption; the shorter one holds its
    last frame until the longer one ends.
    """

    left: list[np.ndarray]
    """
    The frames of the left video.
    """

    right: list[np.ndarray]
    """
    The frames of the right video, of the same height as the left ones.
    """

    left_caption: str
    """
    The caption of the left video.
    """

    right_caption: str
    """
    The caption of the right video.
    """

    @property
    def frames(self) -> list[np.ndarray]:
        """
        :return: The captioned frames of both videos, placed next to each other.
        """
        length = max(len(self.left), len(self.right))
        return [
            np.hstack(
                [
                    self._captioned(
                        self._frame_at(self.left, index), self.left_caption
                    ),
                    self._captioned(
                        self._frame_at(self.right, index), self.right_caption
                    ),
                ]
            )
            for index in range(length)
        ]

    def write(self, path: Path, frames_per_second: int = 30) -> Path:
        """
        :param path: The video file to write.
        :param frames_per_second: The rate the video plays at.
        :return: The path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(path, fps=frames_per_second, macro_block_size=8)
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        return path

    @staticmethod
    def _frame_at(frames: list[np.ndarray], index: int) -> np.ndarray:
        """
        :return: The frame at the index, or the last frame after the video ended.
        """
        return frames[min(index, len(frames) - 1)]

    @staticmethod
    def _captioned(frame: np.ndarray, caption: str) -> np.ndarray:
        """
        :return: The frame with the caption written into its top left corner.
        """
        image = Image.fromarray(frame)
        ImageDraw.Draw(image).text(
            (12, 10), caption, fill=(255, 255, 255), font=ImageFont.load_default(22)
        )
        return np.asarray(image)


# %% comparison of one seed


@dataclass
class ReplanningVideo:
    """
    Videos of one seed performed once as the demo is and once with replanning, and of
    the two side by side.
    """

    output_directory: Path
    """
    Directory the videos and the files of both trials are written to.
    """

    seed: int
    """
    The seed both trials are run with.
    """

    illegal_joint_states: list[Path] = field(default_factory=list)
    """
    The databases the replanning trial replans around.
    """

    collision_avoidance: bool = False
    """
    Whether Giskard avoids collisions in both trials.
    """

    replans_on_stagnation: bool = False
    """
    Whether the replanning trial replans motions that stop approaching their target,
    instead of around :attr:`illegal_joint_states`.
    """

    @staticmethod
    def first_improved_seed(
        baseline: ExperimentOutcome, replanning: ExperimentOutcome
    ) -> int:
        """
        :return: The lowest seed that completed without any collision with replanning
            and did not without it.
        """
        return min(replanning.successful_seeds - baseline.successful_seeds)

    @property
    def path(self) -> Path:
        """
        :return: The side by side video.
        """
        return self.output_directory / f"daisy_cable_replanning_seed{self.seed}.mp4"

    def record(self) -> Path:
        """
        Run both trials with video, then write them side by side.

        :return: The side by side video.
        """
        without = self._trial("without_replanning", replans=False)
        with_replanning = self._trial("with_replanning", replans=True)
        for trial in (without, with_replanning):
            trial.run_in_own_process()
        return SideBySideVideo(
            left=self._frames_of(without.video),
            right=self._frames_of(with_replanning.video),
            left_caption=(
                VideoCaption.WITH_COLLISION_AVOIDANCE
                if self.collision_avoidance
                else VideoCaption.WITHOUT_REPLANNING
            ),
            right_caption=(
                VideoCaption.WITH_STAGNATION_REPLANNING
                if self.replans_on_stagnation
                else VideoCaption.WITH_REPLANNING
            ),
        ).write(self.path)

    @staticmethod
    def _frames_of(video: Path) -> list[np.ndarray]:
        """
        :return: The frames of the video file.
        """
        # Not list(reader): the reader reports an infinite length for MP4 files.
        return [frame for frame in imageio.get_reader(video)]

    def _trial(self, name: str, replans: bool) -> CableDemoTrial:
        """
        :param name: Name of the directory the trial writes its files and video to.
        :param replans: Whether the trial replans.
        :return: The trial of the seed.
        """
        directory = self.output_directory / name
        return CableDemoTrial(
            number=0,
            seed=self.seed,
            files=TrialFiles(directory=directory, number=0),
            illegal_joint_states=self.illegal_joint_states if replans else [],
            replans_on_stagnation=replans and self.replans_on_stagnation,
            collision_avoidance=self.collision_avoidance,
            video=directory / f"{name}.mp4",
        )
