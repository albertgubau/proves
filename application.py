from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QLabel, QFileDialog, QLineEdit, QTabWidget
from PyQt5 import uic
import sys

from function import *

class UI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the ui file created with Qt Designer
        uic.loadUi("prova.ui", self)

        configure_dft_tab(self)

        # Show the App
        self.show()


# Initialize the app
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
