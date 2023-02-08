
from PyQt5 import uic
import sys

from PyQt5.QtCore import Qt

from pyqtgraph.Qt import QtCore

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

def configure_rt_sine_tab(tab):

    layout = QHBoxLayout()

    # Add Widgets to the Layout
    tab.slider = Slider(0, 2)
    layout.addWidget(tab.slider)
    # Set the Layout on the application window
    tab.setLayout(layout)

    pg.setConfigOptions(antialias=True)
    tab.win = pg.GraphicsLayoutWidget()
    tab.win.setWindowTitle('Spectrum Analyzer')
    layout.addWidget(tab.win)

    tab.traces = dict()

    tab.y = []
    tab.frames = []
    tab.result = np.array(0)
    tab.results = np.array([])

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
    tab.waveform = tab.win.addPlot(
        title='WAVEFORM', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
    )

    # Add plots to the window
    tab.w_waveform = tab.win.addPlot(
        title='Windowed WAVEFORM', row=2, col=1, axisItems={'bottom': wf_w_xaxis, 'left': wf_w_yaxis},
    )

    tab.spectrum = tab.win.addPlot(
        title='SPECTRUM', row=3, col=1, axisItems={'bottom': sp_xaxis},
    )

    tab.out = tab.win.addPlot(
        title='OUT', row=4, col=1, axisItems={'bottom': out_xaxis, 'left': out_yaxis},
    )

    tab.iterations = 0
    tab.wf_data = np.array([])

    tab.prova = np.array([])

    # PyAudio Stuff
    tab.FORMAT = pyaudio.paFloat32
    tab.CHANNELS = 1  # Mono
    tab.RATE = 44100  # Sampling rate in Hz (samples/second)
    tab.CHUNK = 2048  # Number of samples per frame (audio frame with frameSize = 2048)

    tab.p = pyaudio.PyAudio()  # Instance pyAudio class

    tab.stream = tab.p.open(  # Create the data stream with the previous parameters
        format=tab.FORMAT,
        channels=tab.CHANNELS,
        rate=tab.RATE,
        input=True,
        output=True,
        frames_per_buffer=tab.CHUNK,
    )

    # Waveform and Spectrum x-axis points (bins and Hz)
    tab.freqs = np.arange(0, tab.CHUNK)
    # Waveform and Spectrum x-axis points (bins and Hz)
    tab.z = np.arange(0, tab.CHUNK)
    # Waveform and Spectrum x-axis points (bins and Hz)
    tab.j = np.arange(0, 512)
    # Half spectrum because of essentia computation
    tab.f = np.linspace(0, tab.RATE // 2, tab.CHUNK // 2 + 1)  # 1025 numbers from 0 to 22050 (frequencies)


def set_plotdata(tab, name, data_x, data_y):

    if name in tab.traces:
        tab.traces[name].setData(data_x, data_y)

    else:
        if name == 'waveform':
            tab.traces[name] = tab.waveform.plot(pen='c', width=3)
            tab.waveform.setYRange(-0.05, 0.05, padding=0)
            tab.waveform.setXRange(0, tab.CHUNK, padding=0.005)

        if name == 'w_waveform':
            tab.traces[name] = tab.w_waveform.plot(pen='c', width=3)
            tab.w_waveform.setYRange(-5e-5, 5e-5, padding=0)
            tab.w_waveform.setXRange(0, tab.CHUNK, padding=0.005)

        if name == 'spectrum':
            tab.traces[name] = tab.spectrum.plot(pen='m', width=3)
            tab.spectrum.setLogMode(x=True, y=True)
            tab.spectrum.setYRange(np.log10(0.001), np.log10(20), padding=0)
            tab.spectrum.setXRange(np.log10(20), np.log10(tab.RATE / 2), padding=0.005)

        if name == 'out':
            tab.traces[name] = tab.out.plot(pen='c', width=3)
            tab.out.setYRange(-0.02, 0.02, padding=0)
            tab.out.setXRange(0, tab.CHUNK // 4, padding=0.005)


def update_plots(tab):

        previous_wf_data1 = tab.wf_data[511:2048]
        previous_wf_data2 = tab.wf_data[1023:2048]
        previous_wf_data3 = tab.wf_data[1535:2048]

        # Get the data from the mic
        tab.wf_data = tab.stream.read(tab.CHUNK)

        # Unpack the data as ints
        tab.wf_data = np.array(
            struct.unpack(str(tab.CHUNK) + 'f', tab.wf_data))  # str(self.CHUNK) + 'h' denotes size and type of data

        tab.prova = np.append(tab.prova, tab.wf_data)

        # Aqui hem llegit un frame de 2048 samples provinent del micro, el plotegem
        set_plotdata(tab,name='waveform', data_x=tab.freqs, data_y=tab.wf_data)

        # Li apliquem windowing i ho plotegem
        set_plotdata(tab,name='w_waveform', data_x=tab.z, data_y=w(tab.wf_data))

        # Apliquem la fft al windowed frame
        fft_signal = fft(w(tab.wf_data))

        # Sine Analysis to get tfreq for the current frame
        sine_anal = sineAnal(fft_signal)  # li entra una fft de 1025 samples

        # Frequency scaling values
        freqScaling = 1.5
        print(tab.slider.x)
        ysfreq = sine_anal[0] * tab.slider.x  # scale of frequencies

        # Synthesis (with OverlapAdd and IFFT)
        fft_synth = sineSynth(sine_anal[1], ysfreq, sine_anal[2])  # retorna un frame de 1025 samples

        sp_data = np.abs(fft(tab.wf_data))

        set_plotdata(tab, name='spectrum', data_x=tab.f, data_y=sp_data)

        if tab.iterations != 0:
            # First auxiliary waveform
            wf_data1 = np.append(previous_wf_data1, tab.wf_data[1:512])

            fft1 = fft(w(wf_data1))
            sine_anal1 = sineAnal(fft1)
            ysfreq1 = sine_anal1[0] * tab.slider.x
            fft_synth1 = sineSynth(sine_anal1[1], ysfreq1, sine_anal1[2])  # retorna un frame de 1025 samples

            out1 = overl(ifft(fft_synth1))  # Tenim un frame de 512 samples

            # Second auxiliary waveform
            wf_data2 = np.append(previous_wf_data2, tab.wf_data[1:1024])

            fft2 = fft(w(wf_data2))
            sine_anal2 = sineAnal(fft2)
            ysfreq2 = sine_anal2[0] * tab.slider.x
            fft_synth2 = sineSynth(sine_anal2[1], ysfreq2, sine_anal2[2])  # retorna un frame de 1025 samples

            out2 = overl(ifft(fft_synth2))  # Tenim un frame de 512 samples

            # Third auxiliary waveform
            wf_data3 = np.append(previous_wf_data3, tab.wf_data[1:1536])

            fft3 = fft(w(wf_data3))
            sine_anal3 = sineAnal(fft3)
            ysfreq3 = sine_anal3[0] * tab.slider.x
            fft_synth3 = sineSynth(sine_anal3[1], ysfreq3, sine_anal3[2])  # retorna un frame de 1025 samples

            out3 = overl(ifft(fft_synth3))  # Tenim un frame de 512 samples

            tab.results = np.append(np.append(out1, out2), out3)
            tab.result = np.append(tab.result, tab.results)

        out = overl(ifft(fft_synth))  # Tenim un frame de 512 samples

        set_plotdata(tab, name='out', data_x=tab.j, data_y=out)

        # Save result and play it simultaneously
        tab.result = np.append(tab.result, out)

        # We cut the signal to not lag the program with large arrays
        # if(len(self.result)>=4097):
        # self.result = self.result[len(self.result) - 4096:]

        sd.play(tab.result[len(tab.result) - 4096:], 44100)
        time.sleep(0.01)
        tab.iterations = 1

def animation(tab):
    timer = QtCore.QTimer()
    timer.timeout.connect(tab)
    timer.start(20)

def getResult(self):
    return self.rt_sine_trans_tab.result()

def getProva(self):
    return self.rt_sine_trans_tab.prova()

def getTab(self):
    return self.rt_sine_trans_tab