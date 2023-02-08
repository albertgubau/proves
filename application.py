from PyQt5 import uic
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

from pyqtgraph.Qt import QtCore
from function import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import essentia.standard as es
import struct
import pyaudio
import sounddevice as sd
import sys
import time


# Instantiate the Essentia Algorithms
w = es.Windowing(type='hamming', size=2048)

fft = es.FFT(size=2048)

sineAnal = es.SineModelAnal(sampleRate=44100,
                            maxnSines=150,
                            magnitudeThreshold=-120,
                            freqDevOffset=10,
                            freqDevSlope=0.001)

sineSynth = es.SineModelSynth(sampleRate=44100, fftSize=2048, hopSize=522)

ifft = es.IFFT(size=2048)
overl = es.OverlapAdd(frameSize=2048, hopSize=512)

awrite = es.MonoWriter(filename='output.wav', sampleRate=44100)

awrite2 = es.MonoWriter(filename='prova.wav', sampleRate=44100)

class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QHBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())
    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
                self.maximum - self.minimum)
        self.label.setText("{0:.4g}".format(self.x))

class UI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the ui file created with Qt Designer
        uic.loadUi("prova.ui", self)

        configure_dft_tab(self)

        self.rt_sine_trans_tab = self.findChild(QWidget, "rt_sine_tab")

        self.tabs = self.findChild(QTabWidget, "tabWidget")

        #OLD CODE
        # Create a QHBoxLayout instance
        layout = QHBoxLayout()

        # Add Widgets to the Layout
        self.rt_sine_trans_tab.slider = Slider(0, 2)
        layout.addWidget(self.rt_sine_trans_tab.slider)

        # Set the Layout on the application window
        self.rt_sine_trans_tab.setLayout(layout)

        pg.setConfigOptions(antialias=True)
        self.rt_sine_trans_tab.win = pg.GraphicsLayoutWidget()
        self.rt_sine_trans_tab.win.setWindowTitle('Spectrum Analyzer')
        layout.addWidget(self.rt_sine_trans_tab.win)

        # OLD CODE FROM PREVIOUS APP
        self.rt_sine_trans_tab.traces = dict()

        self.rt_sine_trans_tab.y = []
        self.rt_sine_trans_tab.frames = []
        self.rt_sine_trans_tab.result = np.array(0)
        self.rt_sine_trans_tab.results = np.array([])

        # Waveform x/y axis labels
        wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_xaxis = pg.AxisItem(orientation='bottom')
        wf_xaxis.setTicks([wf_xlabels])
        wf_yaxis = pg.AxisItem(orientation='left')

        # Waveform x/y axis labels
        wf_w_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_w_xaxis = pg.AxisItem(orientation='bottom')
        wf_w_xaxis.setTicks([wf_w_xlabels])
        wf_w_yaxis = pg.AxisItem(orientation='left')

        # Waveform x/y axis labels
        out_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        out_xaxis = pg.AxisItem(orientation='bottom')
        out_xaxis.setTicks([out_xlabels])
        out_yaxis = pg.AxisItem(orientation='left')

        # Spectrum x/y axis labels
        sp_xlabels = [
            (np.log10(10), '10'), (np.log10(100), '100'),
            (np.log10(1000), '1000'), (np.log10(22050), '22050')
        ]
        sp_xaxis = pg.AxisItem(orientation='bottom')
        sp_xaxis.setTicks([sp_xlabels])

        # Add plots to the window
        self.rt_sine_trans_tab.waveform = self.rt_sine_trans_tab.win.addPlot(
            title='WAVEFORM', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
        )

        # Add plots to the window
        self.rt_sine_trans_tab.w_waveform = self.rt_sine_trans_tab.win.addPlot(
            title='Windowed WAVEFORM', row=2, col=1, axisItems={'bottom': wf_w_xaxis, 'left': wf_w_yaxis},
        )

        self.rt_sine_trans_tab.spectrum = self.rt_sine_trans_tab.win.addPlot(
            title='SPECTRUM', row=3, col=1, axisItems={'bottom': sp_xaxis},
        )

        self.rt_sine_trans_tab.out = self.rt_sine_trans_tab.win.addPlot(
            title='OUT', row=4, col=1, axisItems={'bottom': out_xaxis, 'left': out_yaxis},
        )

        self.rt_sine_trans_tab.iterations = 0
        self.rt_sine_trans_tab.wf_data = np.array([])

        self.rt_sine_trans_tab.prova = np.array([])

        # PyAudio Stuff
        self.rt_sine_trans_tab.FORMAT = pyaudio.paFloat32
        self.rt_sine_trans_tab.CHANNELS = 1  # Mono
        self.rt_sine_trans_tab.RATE = 44100  # Sampling rate in Hz (samples/second)
        self.rt_sine_trans_tab.CHUNK = 2048  # Number of samples per frame (audio frame with frameSize = 2048)

        self.rt_sine_trans_tab.p = pyaudio.PyAudio()  # Instance pyAudio class

        self.stream = self.rt_sine_trans_tab.p.open(  # Create the data stream with the previous parameters
            format=self.rt_sine_trans_tab.FORMAT,
            channels=self.rt_sine_trans_tab.CHANNELS,
            rate=self.rt_sine_trans_tab.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.rt_sine_trans_tab.CHUNK,
        )

        # Waveform and Spectrum x-axis points (bins and Hz)
        self.rt_sine_trans_tab.freqs = np.arange(0, self.rt_sine_trans_tab.CHUNK)
        # Waveform and Spectrum x-axis points (bins and Hz)
        self.rt_sine_trans_tab.z = np.arange(0, self.rt_sine_trans_tab.CHUNK)
        # Waveform and Spectrum x-axis points (bins and Hz)
        self.rt_sine_trans_tab.j = np.arange(0, 512)
        # Half spectrum because of essentia computation
        self.rt_sine_trans_tab.f = np.linspace(0, self.rt_sine_trans_tab.RATE // 2, self.rt_sine_trans_tab.CHUNK // 2 + 1)  # 1025 numbers from 0 to 22050 (frequencies)

        # Show the App
        self.show()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()
    def set_plotdata(self, name, data_x, data_y):

        if name in self.rt_sine_trans_tab.traces:
            self.rt_sine_trans_tab.traces[name].setData(data_x, data_y)

        else:
            if name == 'waveform':
                self.rt_sine_trans_tab.traces[name] = self.rt_sine_trans_tab.waveform.plot(pen='c', width=3)
                self.rt_sine_trans_tab.waveform.setYRange(-0.05, 0.05, padding=0)
                self.rt_sine_trans_tab.waveform.setXRange(0, self.rt_sine_trans_tab.CHUNK, padding=0.005)

            if name == 'w_waveform':
                self.rt_sine_trans_tab.traces[name] = self.rt_sine_trans_tab.w_waveform.plot(pen='c', width=3)
                self.rt_sine_trans_tab.w_waveform.setYRange(-5e-5, 5e-5, padding=0)
                self.rt_sine_trans_tab.w_waveform.setXRange(0, self.CHUNK, padding=0.005)

            if name == 'spectrum':
                self.rt_sine_trans_tab.traces[name] = self.rt_sine_trans_tab.spectrum.plot(pen='m', width=3)
                self.rt_sine_trans_tab.spectrum.setLogMode(x=True, y=True)
                self.rt_sine_trans_tab.spectrum.setYRange(np.log10(0.001), np.log10(20), padding=0)
                self.rt_sine_trans_tab.spectrum.setXRange(np.log10(20), np.log10(self.rt_sine_trans_tab.RATE / 2), padding=0.005)

            if name == 'out':
                self.rt_sine_trans_tab.traces[name] = self.rt_sine_trans_tab.out.plot(pen='c', width=3)
                self.rt_sine_trans_tab.out.setYRange(-0.02, 0.02, padding=0)
                self.rt_sine_trans_tab.out.setXRange(0, self.rt_sine_trans_tab.CHUNK // 4, padding=0.005)
    def update_plots(self):

        previous_wf_data1 = self.rt_sine_trans_tab.wf_data[511:2048]
        previous_wf_data2 = self.rt_sine_trans_tab.wf_data[1023:2048]
        previous_wf_data3 = self.rt_sine_trans_tab.wf_data[1535:2048]

        # Get the data from the mic
        self.rt_sine_trans_tab.wf_data = self.stream.read(self.rt_sine_trans_tab.CHUNK)

        # Unpack the data as ints
        self.rt_sine_trans_tab.wf_data = np.array(
            struct.unpack(str(self.rt_sine_trans_tab.CHUNK) + 'f', self.rt_sine_trans_tab.wf_data))  # str(self.CHUNK) + 'h' denotes size and type of data

        self.rt_sine_trans_tab.prova = np.append(self.rt_sine_trans_tab.prova, self.rt_sine_trans_tab.wf_data)

        # Aqui hem llegit un frame de 2048 samples provinent del micro, el plotegem
        self.set_plotdata(name='waveform', data_x=self.rt_sine_trans_tab.freqs, data_y=self.rt_sine_trans_tab.wf_data)

        # Li apliquem windowing i ho plotegem
        self.set_plotdata(name='w_waveform', data_x=self.rt_sine_trans_tab.z, data_y=w(self.rt_sine_trans_tab.wf_data))

        # Apliquem la fft al windowed frame
        fft_signal = fft(w(self.rt_sine_trans_tab.wf_data))

        # Sine Analysis to get tfreq for the current frame
        sine_anal = sineAnal(fft_signal)  # li entra una fft de 1025 samples

        # Frequency scaling values
        freqScaling = 1.5
        print(self.rt_sine_trans_tab.slider.x)
        ysfreq = sine_anal[0] * self.rt_sine_trans_tab.slider.x  # scale of frequencies

        # Synthesis (with OverlapAdd and IFFT)
        fft_synth = sineSynth(sine_anal[1], ysfreq, sine_anal[2])  # retorna un frame de 1025 samples

        sp_data = np.abs(fft(self.rt_sine_trans_tab.wf_data))

        self.set_plotdata(name='spectrum', data_x=self.rt_sine_trans_tab.f, data_y=sp_data)

        if self.rt_sine_trans_tab.iterations != 0:
            # First auxiliary waveform
            wf_data1 = np.append(previous_wf_data1, self.rt_sine_trans_tab.wf_data[1:512])

            fft1 = fft(w(wf_data1))
            sine_anal1 = sineAnal(fft1)
            ysfreq1 = sine_anal1[0] * self.rt_sine_trans_tab.slider.x
            fft_synth1 = sineSynth(sine_anal1[1], ysfreq1, sine_anal1[2])  # retorna un frame de 1025 samples

            out1 = overl(ifft(fft_synth1))  # Tenim un frame de 512 samples

            # Second auxiliary waveform
            wf_data2 = np.append(previous_wf_data2, self.rt_sine_trans_tab.wf_data[1:1024])

            fft2 = fft(w(wf_data2))
            sine_anal2 = sineAnal(fft2)
            ysfreq2 = sine_anal2[0] * self.rt_sine_trans_tab.slider.x
            fft_synth2 = sineSynth(sine_anal2[1], ysfreq2, sine_anal2[2])  # retorna un frame de 1025 samples

            out2 = overl(ifft(fft_synth2))  # Tenim un frame de 512 samples

            # Third auxiliary waveform
            wf_data3 = np.append(previous_wf_data3, self.rt_sine_trans_tab.wf_data[1:1536])

            fft3 = fft(w(wf_data3))
            sine_anal3 = sineAnal(fft3)
            ysfreq3 = sine_anal3[0] * self.rt_sine_trans_tab.slider.x
            fft_synth3 = sineSynth(sine_anal3[1], ysfreq3, sine_anal3[2])  # retorna un frame de 1025 samples

            out3 = overl(ifft(fft_synth3))  # Tenim un frame de 512 samples

            self.rt_sine_trans_tab.results = np.append(np.append(out1, out2), out3)
            self.rt_sine_trans_tab.result = np.append(self.rt_sine_trans_tab.result, self.rt_sine_trans_tab.results)

        out = overl(ifft(fft_synth))  # Tenim un frame de 512 samples

        self.set_plotdata(name='out', data_x=self.rt_sine_trans_tab.j, data_y=out)

        # Save result and play it simultaneously
        self.rt_sine_trans_tab.result = np.append(self.rt_sine_trans_tab.result, out)

        # We cut the signal to not lag the program with large arrays
        # if(len(self.result)>=4097):
        # self.result = self.result[len(self.result) - 4096:]

        sd.play(self.rt_sine_trans_tab.result[len(self.rt_sine_trans_tab.result) - 4096:], 44100)
        time.sleep(0.01)
        self.rt_sine_trans_tab.iterations = 1
    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update_plots)
        timer.start(20)
        self.start()
    def saveResult(self):
        awrite(self.rt_sine_trans_tab.result)
        awrite2(self.rt_sine_trans_tab.prova)

# Initialize the app
app = QApplication(sys.argv)
UIWindow = UI()
UIWindow.animation()
UIWindow.saveResult()
