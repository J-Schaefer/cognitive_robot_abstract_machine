from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QToolBar,
    QFileDialog,
    QStackedWidget,
)
from PySide6.QtGui import QAction

from .controller import ModelController
from .home_widget import HomeWidget
from .query_widget import QueryWidget
from .posterior_widget import PosteriorWidget
from .mode_widget import ModeWidget


class MainWindow(QMainWindow):
    """
    Main Window of the Probabilistic Model GUI.
    """

    controller: ModelController
    home_widget: HomeWidget
    query_widget: QueryWidget
    posterior_widget: PosteriorWidget
    mode_widget: ModeWidget
    central_stack: QStackedWidget

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Probabilistic Model GUI")
        self.resize(1000, 700)

        self.controller = ModelController()
        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface.
        """
        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        load_action = QAction("Load Model", self)
        load_action.triggered.connect(self.load_model)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        home_action = QAction("Home", self)
        home_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.home_widget)
        )
        toolbar.addAction(home_action)

        query_action = QAction("Query", self)
        query_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.query_widget)
        )
        toolbar.addAction(query_action)

        posterior_action = QAction("Posterior", self)
        posterior_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.posterior_widget)
        )
        toolbar.addAction(posterior_action)

        mode_action = QAction("Mode", self)
        mode_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.mode_widget)
        )
        toolbar.addAction(mode_action)

        # Central Widget
        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        self.home_widget = HomeWidget(self.controller)
        self.query_widget = QueryWidget(self.controller)
        self.posterior_widget = PosteriorWidget(self.controller)
        self.mode_widget = ModeWidget(self.controller)

        self.central_stack.addWidget(self.home_widget)
        self.central_stack.addWidget(self.query_widget)
        self.central_stack.addWidget(self.posterior_widget)
        self.central_stack.addWidget(self.mode_widget)

    def load_model(self):
        """
        Opens a file dialog to load a model.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("JSON Files (*.json)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.controller.load_model_from_json_file(file_paths[0])
                self.home_widget.refresh_variable_list()
                # Re-initialize query widget or refresh it
                self.central_stack.removeWidget(self.query_widget)
                self.query_widget.deleteLater()
                self.query_widget = QueryWidget(self.controller)
                self.central_stack.addWidget(self.query_widget)

                self.central_stack.removeWidget(self.posterior_widget)
                self.posterior_widget.deleteLater()
                self.posterior_widget = PosteriorWidget(self.controller)
                self.central_stack.addWidget(self.posterior_widget)

                self.central_stack.removeWidget(self.mode_widget)
                self.mode_widget.deleteLater()
                self.mode_widget = ModeWidget(self.controller)
                self.central_stack.addWidget(self.mode_widget)
