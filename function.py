from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QLabel, QFileDialog, QLineEdit, QTabWidget
from PyQt5 import uic
import sys


def configure_dft_tab(self):
    self.dft_tab = self.findChild(QWidget, "dft_tab")
    self.browse_button = self.dft_tab.findChild(QPushButton, "browse_btn")
    self.input_text_box = self.findChild(QLineEdit, "filename")
    self.browse_button.clicked.connect(lambda: browse_file(self))


def browse_file(self):
    # self.title_label.setText("You Clicked the button!") # We should modifiy the input text box
    # Open File Dialog (returns a tuple)
    fname = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")

    # Output filename to screen
    if fname:
        self.input_text_box.setText(str(fname[0]))  # We should modifiy the input text box
        # print(self.input_text_box.text()) # Get the text from the text box (i.e, path to the input file)
