"""
The window that lists recorded illegal joint states, shows them with the robot and keeps
chosen ones in a permanent file.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from experiments.daisy_cable_replanning.collisions import (
    CollisionKind,
    IllegalJointState,
    IllegalJointStateDatabase,
    MotionLabel,
)
from experiments.daisy_cable_replanning.joint_state_browser import (
    JointStateBrowser,
    StateColumn,
)
from experiments.daisy_cable_replanning.robot_snapshot import RobotSnapshot
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from semantic_digital_twin.utils import daisy_installed

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")

# %% fixtures


@pytest.fixture(scope="session")
def application() -> Iterator[QApplication]:
    """
    The Qt application, drawing to no screen.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        yield QApplication.instance() or QApplication([])


def _state(positions: np.ndarray, trial: int) -> IllegalJointState:
    """
    :return: An illegal joint state of both DAiSy arms at the positions.
    """
    joint_names = IllegalJointStateDatabase.for_daisy_arms().joint_names
    return IllegalJointState(
        positions=positions.tolist(),
        colliding_joint_names=joint_names[6:],
        collision_kind=CollisionKind.WITHIN_ARM,
        first_body="right_forearm_link",
        second_body="right_gripper_side_cylinder_link",
        penetration_depth=0.019,
        motion=MotionLabel(
            stage="rehang", index=1, action="CableRehangAction", motion="Move"
        ),
        trial=trial,
    )


@pytest.fixture()
def recorded_states(
    right_forearm_touching_gripper: np.ndarray,
) -> list[IllegalJointState]:
    """
    Two recorded states: the right forearm in its gripper, and the same with the right
    wrist turned away.
    """
    turned = right_forearm_touching_gripper.copy()
    turned[11] += 1.0
    return [_state(right_forearm_touching_gripper, 0), _state(turned, 1)]


@pytest.fixture()
def files(
    tmp_path: Path, recorded_states: list[IllegalJointState]
) -> tuple[Path, Path]:
    """
    A recorded file of both states, and a permanent file holding the second.
    """
    recorded = IllegalJointStateDatabase.for_daisy_arms()
    recorded.illegal_joint_states.extend(recorded_states)
    recorded.save(tmp_path / "recorded.json")
    kept = IllegalJointStateDatabase.for_daisy_arms()
    kept.illegal_joint_states.append(recorded_states[1])
    kept.save(tmp_path / "permanent.json")
    return tmp_path / "recorded.json", tmp_path / "permanent.json"


@pytest.fixture()
def window(
    application: QApplication, files: tuple[Path, Path]
) -> Iterator[JointStateBrowser]:
    """
    The window, with both files loaded.
    """
    recorded, permanent = files
    browser = JointStateBrowser(recorded, permanent)
    yield browser
    browser.robot_view.snapshot.stop()
    browser.close()


# %% listing and choosing


def test_window_lists_the_recorded_states(window: JointStateBrowser):
    assert window.table.rowCount() == 2


def test_states_in_the_permanent_file_start_out_kept(window: JointStateBrowser):
    keep = [
        window.table.data(window.table.index(row, StateColumn.KEEP), Qt.CheckStateRole)
        for row in range(window.table.rowCount())
    ]

    assert keep == [Qt.Unchecked, Qt.Checked]


def test_saving_writes_the_kept_states_to_the_permanent_file(
    window: JointStateBrowser,
    files: tuple[Path, Path],
    recorded_states: list[IllegalJointState],
):
    window.keep_all(False)
    window.table.setData(
        window.table.index(0, StateColumn.KEEP), Qt.Checked, Qt.CheckStateRole
    )

    window.save()

    _, permanent = files
    saved = IllegalJointStateDatabase.load(permanent)
    assert saved.illegal_joint_states == [recorded_states[0]]


# %% showing


def test_clicking_a_state_shows_it_with_the_robot(
    window: JointStateBrowser, application: QApplication
):
    window.view.selectRow(0)
    deadline = time.monotonic() + 10
    while window.robot_view.pixmap().isNull() and time.monotonic() < deadline:
        application.processEvents()

    assert not window.robot_view.pixmap().isNull()
    assert "right_forearm_link" in window.details.text()


def test_snapshot_finds_the_contact_of_the_state_shown(
    right_forearm_touching_gripper: np.ndarray,
):
    snapshot = RobotSnapshot.of_cable_workcell(
        IllegalJointStateDatabase.for_daisy_arms().joint_names
    )
    contacts_parked = snapshot.contacts()
    snapshot.show(right_forearm_touching_gripper.tolist())
    contacts_shown = snapshot.contacts()
    image = snapshot.render(snapshot.overview(), width=320, height=240)
    snapshot.stop()

    assert contacts_parked == []
    assert {
        frozenset((contact.body_a.name.name, contact.body_b.name.name))
        for contact in contacts_shown
    } == {frozenset(("right_forearm_link", "right_gripper_side_cylinder_link"))}
    assert image.shape == (240, 320, 3)
