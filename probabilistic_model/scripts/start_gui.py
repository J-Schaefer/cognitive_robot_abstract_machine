import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from probabilistic_model.gui.main_window import MainWindow
from probabilistic_model.learning.jpt.variables import *  # type: ignore


def main():
    """
    Main entry point for starting the Probabilistic Model GUI.
    """
    # Recommended for some Linux environments/Docker
    os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

    # Required for QWebEngineView to work in many environments
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    # Sometimes helps with rendering on certain GPUs/drivers
    # In Qt6, AA_UseDesktopOpenGL is still available but often default.
    # On some systems, AA_UseOpenGLES or AA_UseSoftwareOpenGL might be needed instead.

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
