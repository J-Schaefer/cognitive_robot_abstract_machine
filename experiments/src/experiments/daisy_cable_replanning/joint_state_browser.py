"""
A window listing the illegal joint states of a recording, showing each one with the
robot, and keeping the chosen ones in a permanent database.
"""

from __future__ import annotations

import sys
from enum import IntEnum
from pathlib import Path

import numpy as np
from experiments.daisy_cable_replanning.collisions import IllegalJointState
from experiments.daisy_cable_replanning.curation import IllegalJointStateCuration
from experiments.daisy_cable_replanning.robot_snapshot import (
    OrbitViewpoint,
    RobotSnapshot,
)
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPoint,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

# %% table of the recorded states


class StateColumn(IntEnum):
    """
    The columns of the table of recorded illegal joint states, in order.
    """

    KEEP = 0
    KEPT_ALREADY = 1
    TRIAL = 2
    MOTION = 3
    COLLISION_KIND = 4
    BODIES = 5
    PENETRATION_DEPTH = 6

    @property
    def header(self) -> str:
        """
        :return: The title of the column.
        """
        match self:
            case StateColumn.KEEP:
                return "Keep"
            case StateColumn.KEPT_ALREADY:
                return "In permanent file"
            case StateColumn.TRIAL:
                return "Trial"
            case StateColumn.MOTION:
                return "Motion"
            case StateColumn.COLLISION_KIND:
                return "Collision"
            case StateColumn.BODIES:
                return "Bodies"
            case StateColumn.PENETRATION_DEPTH:
                return "Depth [mm]"


class IllegalJointStateTable(QAbstractTableModel):
    """
    The recorded illegal joint states of a curation, one per row, with a check box per
    row choosing whether it is kept.
    """

    def __init__(self):
        super().__init__()
        self.curation: IllegalJointStateCuration | None = None
        """
        The curation shown, or ``None`` before a recording is loaded.
        """

    def show_curation(self, curation: IllegalJointStateCuration) -> None:
        """
        Show the recorded states of the curation.
        """
        self.beginResetModel()
        self.curation = curation
        self.endResetModel()

    def state_at(self, row: int) -> IllegalJointState:
        """
        :return: The recorded state of the row.
        """
        return self.curation.recorded.illegal_joint_states[row]

    def set_kept(self, rows: list[int], kept: bool) -> None:
        """
        Choose or unchoose the recorded states of the rows.
        """
        for row in rows:
            self.curation.chosen[row] = kept
        self.dataChanged.emit(
            self.index(0, StateColumn.KEEP),
            self.index(self.rowCount() - 1, StateColumn.KEEP),
        )

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if self.curation is None or (parent is not None and parent.isValid()):
            return 0
        return len(self.curation.recorded.illegal_joint_states)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(StateColumn)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> str | None:
        if orientation != Qt.Horizontal or role != Qt.DisplayRole:
            return None
        return StateColumn(section).header

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == StateColumn.KEEP:
            flags |= Qt.ItemIsUserCheckable
        return flags

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        state = self.state_at(index.row())
        column = StateColumn(index.column())
        if role == Qt.CheckStateRole and column == StateColumn.KEEP:
            return Qt.Checked if self.curation.chosen[index.row()] else Qt.Unchecked
        if role == Qt.UserRole:
            return self._sort_key(index.row(), state, column)
        if role != Qt.DisplayRole:
            return None
        match column:
            case StateColumn.KEEP:
                return None
            case StateColumn.KEPT_ALREADY:
                return "yes" if self.curation.is_kept_already(state) else ""
            case StateColumn.TRIAL:
                return state.trial
            case StateColumn.MOTION:
                return str(state.motion)
            case StateColumn.COLLISION_KIND:
                return state.collision_kind.value
            case StateColumn.BODIES:
                return f"{state.first_body} / {state.second_body}"
            case StateColumn.PENETRATION_DEPTH:
                return f"{state.penetration_depth * 1000:.1f}"

    def setData(
        self, index: QModelIndex, value: object, role: int = Qt.EditRole
    ) -> bool:
        if role != Qt.CheckStateRole or index.column() != StateColumn.KEEP:
            return False
        self.curation.chosen[index.row()] = Qt.CheckState(value) == Qt.Checked
        self.dataChanged.emit(index, index)
        return True

    def _sort_key(
        self, row: int, state: IllegalJointState, column: StateColumn
    ) -> object:
        """
        :return: The value the column sorts the row by.
        """
        match column:
            case StateColumn.KEEP:
                return int(self.curation.chosen[row])
            case StateColumn.KEPT_ALREADY:
                return int(self.curation.is_kept_already(state))
            case StateColumn.TRIAL:
                return state.trial
            case StateColumn.PENETRATION_DEPTH:
                return state.penetration_depth
            case _:
                return self.data(self.index(row, column))


# %% robot view


class RobotView(QLabel):
    """
    An image of the robot in the joint state shown; dragging orbits the camera, the
    mouse wheel zooms.
    """

    def __init__(self):
        super().__init__("Load a recorded file to see its joint states.")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        # Ignored, so the rendered pixmap does not grow the view it is rendered for.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.snapshot: RobotSnapshot | None = None
        """
        The robot rendered, built once the joints of a recording are known.
        """
        self.viewpoint: OrbitViewpoint | None = None
        """
        Where the camera looks from.
        """
        self._last_mouse_position: QPoint | None = None
        """
        Where the mouse was while dragging, or ``None`` while not dragging.
        """
        self._render_timer = QTimer(self)
        """
        Gathers render requests that come in quick succession into one rendering.
        """
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(30)
        self._render_timer.timeout.connect(self._render)

    def show_joints(self, joint_names: list[str]) -> None:
        """
        Prepare the robot whose joints a recording gives positions of.
        """
        if (
            self.snapshot is not None
            and [connection.name.name for connection in self.snapshot.connections]
            == joint_names
        ):
            return
        if self.snapshot is not None:
            self.snapshot.stop()
        self.snapshot = RobotSnapshot.of_cable_workcell(joint_names)
        self.viewpoint = self.snapshot.overview()

    def show_state(self, state: IllegalJointState) -> None:
        """
        Show the robot in the joint state.
        """
        self.snapshot.show(state.positions)
        self.request_render()

    def focus_on_deepest_contact(self) -> None:
        """
        Look closely at the deepest contact of the joint state shown, if it has one.
        """
        contacts = self.snapshot.contacts()
        if not contacts:
            return
        self.viewpoint = self.viewpoint.focused_on(
            contacts[0].root_P_point_on_body_a[:3], distance=0.6
        )
        self.request_render()

    def reset_viewpoint(self) -> None:
        """
        Look at the whole scene again.
        """
        self.viewpoint = self.snapshot.overview()
        self.request_render()

    def request_render(self) -> None:
        """
        Render the robot again soon.
        """
        if self.snapshot is not None:
            self._render_timer.start()

    def _render(self) -> None:
        """
        Render the robot at the aspect ratio of the view, at most as large as MuJoCo
        renders, and scale the image to the view.
        """
        scale = min(
            1.0,
            self.snapshot.largest_image_width / max(1, self.width()),
            self.snapshot.largest_image_height / max(1, self.height()),
        )
        # MuJoCo renders widths that are multiples of four without padding.
        width = max(4, int(self.width() * scale) // 4 * 4)
        height = max(4, int(self.height() * scale))
        image = np.ascontiguousarray(
            self.snapshot.render(self.viewpoint, width, height)
        )
        pixmap = QPixmap.fromImage(
            QImage(image.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        )
        self.setPixmap(
            pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_position = event.position().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_position = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse_position is None or self.viewpoint is None:
            return
        position = event.position().toPoint()
        movement = position - self._last_mouse_position
        self._last_mouse_position = position
        self.viewpoint = self.viewpoint.orbited(
            -0.4 * movement.x(), -0.4 * movement.y()
        )
        self.request_render()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.viewpoint is None:
            return
        self.viewpoint = self.viewpoint.zoomed(
            0.9 if event.angleDelta().y() > 0 else 1.1
        )
        self.request_render()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.request_render()


# %% file fields


class FileField(QWidget):
    """
    A path entered as text or chosen in a file dialog.
    """

    def __init__(self, title: str, path: Path | None, must_exist: bool):
        super().__init__()
        self.title = title
        """
        What the file is, shown in the file dialog.
        """
        self.must_exist = must_exist
        """
        Whether only an existing file can be chosen.
        """
        self.line = QLineEdit("" if path is None else str(path))
        """
        The path as text.
        """
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line)
        layout.addWidget(browse)

    @property
    def path(self) -> Path:
        """
        :return: The path entered.
        """
        return Path(self.line.text()).expanduser()

    def _browse(self) -> None:
        """
        Let the user choose the file in a dialog.
        """
        if self.must_exist:
            path, _ = QFileDialog.getOpenFileName(
                self, self.title, self.line.text(), "JSON (*.json)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self,
                self.title,
                self.line.text(),
                "JSON (*.json)",
                options=QFileDialog.DontConfirmOverwrite,
            )
        if path:
            self.line.setText(path)


# %% window


class JointStateBrowser(QMainWindow):
    """
    Lists the illegal joint states of a recorded file, shows the one clicked with the
    robot, and saves the chosen ones into a permanent file.
    """

    def __init__(self, recorded: Path | None, permanent: Path):
        super().__init__()
        self.setWindowTitle("Illegal joint states")
        self.resize(1500, 850)
        self.recorded_field = FileField("Recorded file", recorded, must_exist=True)
        """
        The recorded file the states are listed from.
        """
        self.permanent_field = FileField("Permanent file", permanent, must_exist=False)
        """
        The permanent file the chosen states are saved to.
        """
        self.table = IllegalJointStateTable()
        """
        The recorded states.
        """
        self.robot_view = RobotView()
        """
        The robot in the state clicked.
        """
        self.details = QLabel()
        """
        What is known about the state clicked.
        """
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.view = self._table_view()
        """
        The view of the table.
        """
        self.setCentralWidget(self._layout())
        if recorded is not None:
            self.load()

    @classmethod
    def run(cls, recorded: Path | None, permanent: Path) -> int:
        """
        Open the window and run until it is closed.

        :param recorded: The recorded file to load at once, if any.
        :param permanent: The permanent file to compare with and save to.
        :return: The exit code of the application.
        """
        application = QApplication.instance() or QApplication(sys.argv)
        window = cls(recorded, permanent)
        window.show()
        return application.exec()

    def load(self) -> None:
        """
        Load the recorded file and the permanent file; the recorded states the permanent
        file already holds start out kept.
        """
        curation = IllegalJointStateCuration.load(
            self.recorded_field.path, self.permanent_field.path
        )
        self.statusBar().showMessage("Preparing the robot …")
        QApplication.processEvents()
        self.robot_view.show_joints(curation.recorded.joint_names)
        self.table.show_curation(curation)
        self.view.resizeColumnsToContents()
        self._report_counts("Loaded")
        if self.table.rowCount() > 0:
            self.view.selectRow(0)

    def save(self) -> None:
        """
        Save the kept states into the permanent file, and load both files again so the
        table shows what the permanent file now holds.
        """
        curated = self.table.curation.curated()
        curated.save(self.permanent_field.path)
        self.load()
        self.statusBar().showMessage(
            f"Saved {len(curated.illegal_joint_states)} joint states to "
            f"{self.permanent_field.path}"
        )

    def keep_all(self, kept: bool) -> None:
        """
        Keep, or keep none of, the recorded states.
        """
        self.table.set_kept(list(range(self.table.rowCount())), kept)
        self._report_counts("Changed")

    def keep_highlighted(self, kept: bool) -> None:
        """
        Keep, or do not keep, the recorded states of the highlighted rows.
        """
        self.table.set_kept(self._highlighted_rows(), kept)
        self._report_counts("Changed")

    def _highlighted_rows(self) -> list[int]:
        """
        :return: The rows of the table, in the order of the recording, that are
            highlighted.
        """
        proxy = self.view.model()
        return sorted(
            proxy.mapToSource(index).row()
            for index in self.view.selectionModel().selectedRows()
        )

    def _show_clicked(self, current: QModelIndex, _: QModelIndex) -> None:
        """
        Show the state of the row clicked with the robot, and its details.
        """
        if not current.isValid():
            return
        state = self.table.state_at(self.view.model().mapToSource(current).row())
        self.robot_view.show_state(state)
        contacts = self.robot_view.snapshot.contacts()
        joints = ", ".join(
            f"{name.removesuffix('_joint')} {position:.3f}"
            for name, position in zip(
                self.table.curation.recorded.joint_names, state.positions
            )
        )
        touching = "; ".join(
            f"{contact.body_a.name.name} / {contact.body_b.name.name} "
            f"{-contact.distance * 1000:.1f} mm"
            for contact in contacts
        )
        self.details.setText(
            f"<b>{state.motion}</b>, trial {state.trial}<br>"
            f"{state.collision_kind.value}: {state.first_body} / {state.second_body}, "
            f"{state.penetration_depth * 1000:.1f} mm deep when recorded<br>"
            f"Touching now, next to the nominal workcell: {touching or 'nothing'}<br>"
            f"<small>{joints}</small>"
        )

    def _report_counts(self, verb: str) -> None:
        """
        Show how many recorded states there are and how many are kept.
        """
        curation = self.table.curation
        self.statusBar().showMessage(
            f"{verb}: {len(curation.recorded.illegal_joint_states)} recorded joint "
            f"states, {sum(curation.chosen)} kept; the permanent file holds "
            f"{len(curation.kept.illegal_joint_states)}."
        )

    def _table_view(self) -> QTableView:
        """
        :return: The sortable view of the table, showing the state clicked.
        """
        proxy = QSortFilterProxyModel(self)
        proxy.setSourceModel(self.table)
        proxy.setSortRole(Qt.UserRole)
        view = QTableView()
        view.setModel(proxy)
        view.setSortingEnabled(True)
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        view.horizontalHeader().setStretchLastSection(True)
        view.selectionModel().currentRowChanged.connect(self._show_clicked)
        return view

    def _layout(self) -> QWidget:
        """
        :return: The file fields above, the table and the robot side by side, the
            buttons below.
        """
        files = QGridLayout()
        files.addWidget(QLabel("Recorded file"), 0, 0)
        files.addWidget(self.recorded_field, 0, 1)
        files.addWidget(QLabel("Permanent file"), 1, 0)
        files.addWidget(self.permanent_field, 1, 1)
        load = QPushButton("Load")
        load.clicked.connect(self.load)
        files.addWidget(load, 0, 2, 2, 1)

        robot = QWidget()
        robot_layout = QVBoxLayout(robot)
        robot_layout.addWidget(self.robot_view, stretch=1)
        robot_layout.addWidget(self.details)
        view_buttons = QHBoxLayout()
        for title, action in (
            ("Focus on contact", self.robot_view.focus_on_deepest_contact),
            ("Whole scene", self.robot_view.reset_viewpoint),
        ):
            button = QPushButton(title)
            button.clicked.connect(action)
            view_buttons.addWidget(button)
        robot_layout.addLayout(view_buttons)

        splitter = QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(robot)
        splitter.setSizes([800, 700])

        buttons = QHBoxLayout()
        for title, action in (
            ("Keep all", lambda: self.keep_all(True)),
            ("Keep none", lambda: self.keep_all(False)),
            ("Keep highlighted", lambda: self.keep_highlighted(True)),
            ("Drop highlighted", lambda: self.keep_highlighted(False)),
        ):
            button = QPushButton(title)
            button.clicked.connect(action)
            buttons.addWidget(button)
        buttons.addStretch()
        save = QPushButton("Save to permanent file")
        save.clicked.connect(self.save)
        buttons.addWidget(save)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(files)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(buttons)
        return central
