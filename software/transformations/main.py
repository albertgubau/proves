import numpy as np

import essentia.standard as es
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget

import struct
import pyaudio
import wave
import sounddevice as sd
import sys


# Instantiate the Essentia Algorithms
w = es.Windowing(type='hamming', size=2048)

fft = es.FFT(size=2048)

sineAnal = es.SineModelAnal(sampleRate=44100)

sineSynth = es.SineModelSynth(sampleRate=44100, fftSize=2048, hopSize=512)

ifft = es.IFFT(size=2048)
overl = es.OverlapAdd(frameSize=2048, hopSize=512)

awrite = es.MonoWriter(filename='output.wav', sampleRate=44100)
filename = "soundsample.wav"

class AudioStream(object):
    def __init__(self):

        # Initialize the Qt Application
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QApplication(sys.argv)

        self.win = pg.GraphicsLayoutWidget(title='Spectrum Analyzer')
        self.win.setWindowTitle('Spectrum Analyzer')
        self.win.setGeometry(5, 115, 1000, 600)
        self.win.show()

        self.y = []

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
        self.waveform = self.win.addPlot(
            title='WAVEFORM', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
        )

        # Add plots to the window
        self.w_waveform = self.win.addPlot(
            title='Windowed WAVEFORM', row=2, col=1, axisItems={'bottom': wf_w_xaxis, 'left': wf_w_yaxis},
        )

        self.spectrum = self.win.addPlot(
            title='SPECTRUM', row=3, col=1, axisItems={'bottom': sp_xaxis},
        )

        self.out = self.win.addPlot(
            title='OUT', row=4, col=1, axisItems={'bottom': out_xaxis, 'left': out_yaxis},
        )

        self.result = []
        self.iterations = 0
        self.wf_data = np.array([])
        self.previous_wf_data1 = np.array([])
        self.previous_wf_data2 = np.array([])
        self.previous_wf_data3 = np.array([])
        self.wf_data1 = np.array([])
        self.wf_data2 = np.array([])
        self.wf_data3 = np.array([])

        self.filename = "soundsample.wav"

        # PyAudio Stuff
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # Mono
        self.RATE = 44100  # Sampling rate in Hz (samples/second)
        self.CHUNK = 2048  # Number of samples per frame (audio frame with frameSize = 2048)

        self.p = pyaudio.PyAudio()  # Instance pyAudio class

        self.stream = self.p.open(  # Create the data stream with the previous parameters
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )

        # Waveform and Spectrum x-axis points (bins and Hz)
        self.x = np.arange(0, self.CHUNK)
        # Waveform and Spectrum x-axis points (bins and Hz)
        self.z = np.arange(0, self.CHUNK)
        # Waveform and Spectrum x-axis points (bins and Hz)
        self.j = np.arange(0, self.CHUNK // 4)
        # Half spectrum because of essentia computation
        self.f = np.linspace(0, self.RATE // 2, self.CHUNK // 2 + 1)  # 1025 numbers from 0 to 22050 (frequencies)
        self.result = np.array([])


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()

    def set_plotdata(self, name, data_x, data_y):

        if name in self.traces:
            self.traces[name].setData(data_x, data_y)

        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=3)
                self.waveform.setYRange(-60000, 60000, padding=0)
                self.waveform.setXRange(0, self.CHUNK, padding=0.005)

            if name == 'w_waveform':
                self.traces[name] = self.w_waveform.plot(pen='c', width=3)
                # self.w_waveform.setYRange(0, 1, padding=0)
                self.w_waveform.setXRange(0, self.CHUNK, padding=0.005)

            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot(pen='m', width=3)
                self.spectrum.setLogMode(x=True, y=True)
                self.spectrum.setXRange(np.log10(20), np.log10(self.RATE / 2), padding=0.005)

            if name == 'out':
                self.traces[name] = self.out.plot(pen='c', width=3)
                # self.w_waveform.setYRange(0, 1, padding=0)
                self.w_waveform.setXRange(0, self.CHUNK, padding=0.005)

    def update(self):

        self.previous_wf_data1 = self.wf_data[511:2047]
        self.previous_wf_data2 = self.wf_data[1023:2047]
        self.previous_wf_data3 = self.wf_data[1535:2047]

        # Get the data from the mic
        self.wf_data = self.stream.read(self.CHUNK)

        # Unpack the data as ints
        self.wf_data = np.array(struct.unpack(str(self.CHUNK) + 'h', self.wf_data))  # str(self.CHUNK) + 'h' denotes size and type of data

        # Aqui hem llegit un frame de 2048 samples provinent del micro, el plotegem
        self.set_plotdata(name='waveform', data_x=self.x, data_y=self.wf_data)

        # Li apliquem windowing i ho plotegem
        self.set_plotdata(name='w_waveform', data_x=self.z, data_y=w(self.wf_data))

        # Apliquem la fft al windowed frame
        fft_signal = fft(w(self.wf_data))

        # Sine Analysis to get tfreq for the current frame
        sine_anal = sineAnal(fft_signal)  # li entra una fft de 1025 samples

        # Frequency scaling values
        freqScaling = 1

        ysfreq = sine_anal[0] * freqScaling  # scale of frequencies

        # Synthesis (with OverlapAdd and IFFT)
        fft_synth = sineSynth(sine_anal[1], ysfreq, sine_anal[2])  # retorna un frame de 1025 samples

        out = overl(ifft(fft_synth))  # Tenim un frame de 512 samples
        
        outz = out.astype(int)
        self.set_plotdata(name='out', data_x=self.j, data_y=out)

        sp_data = np.abs(fft(self.wf_data))

        self.set_plotdata(name='spectrum', data_x=self.f, data_y=sp_data)

        if self.iterations != 0:

            wf_data1 = np.append(self.previous_wf_data1, self.wf_data[0:512])
            wf_data2 = np.append(self.previous_wf_data2, self.wf_data[0:1024])
            wf_data3 = np.append(self.previous_wf_data3, self.wf_data[0:1536])

            fft1 = fft(w(wf_data1))
            fft2 = fft(w(wf_data2))
            fft3 = fft(w(wf_data3))

            sine_anal1 = sineAnal(fft1)
            sine_anal2 = sineAnal(fft2)
            sine_anal3 = sineAnal(fft3)

            ysfreq1 = sine_anal1[0] * freqScaling  # scale of frequencies
            ysfreq2 = sine_anal2[0] * freqScaling
            ysfreq3 = sine_anal3[0] * freqScaling

            # Synthesis (with OverlapAdd and IFFT)
            fft_synth1 = sineSynth(sine_anal1[1], ysfreq1, sine_anal1[2])  # retorna un frame de 1025 samples
            fft_synth2 = sineSynth(sine_anal2[1], ysfreq2, sine_anal2[2])  # retorna un frame de 1025 samples
            fft_synth3 = sineSynth(sine_anal3[1], ysfreq3, sine_anal3[2])  # retorna un frame de 1025 samples

            out1 = overl(ifft(fft_synth1))  # Tenim un frame de 512 samples
            self.result = np.append(self.result, out1)
            out1 = out1.astype(int)


            repacked1 = struct.pack('<' + str(len(out1)) + 'h', *out1)
            self.y.append(repacked1)


            out2 = overl(ifft(fft_synth2))  # Tenim un frame de 512 samples
            self.result = np.append(self.result, out2)
            out2 = out2.astype(int)


            repacked2 = struct.pack('<' + str(len(out2)) + 'h', *out2)
            self.y.append(repacked2)

            out3 = overl(ifft(fft_synth3))  # Tenim un frame de 512 samples
            self.result = np.append(self.result, out3)
            out3 = out3.astype(int)


            repacked3 = struct.pack('<' + str(len(out3)) + 'h', *out3)
            self.y.append(repacked3)

        # Save result
        self.result = np.append(self.result, out)

        repacked = struct.pack('<' + str(len(out)) + 'h', *outz)
        self.result = np.append(self.result, out)

        self.y.append(repacked)
        self.iterations += 1


    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()

    def saveResult(self):

        # Open and Set the data of the WAV file
        file = wave.open(self.filename, 'wb')
        file.setnchannels(self.CHANNELS)
        # print(self.p.get_sample_size(pyaudio.paInt16))
        file.setsampwidth(2)
        file.setframerate(self.RATE)

        # Write and Close the File
        file.writeframes(b''.join(self.y))
        file.close()
        print(self.result)
        self.result = self.result * (1 / 60000)
        awrite(self.result)


if __name__ == '__main__':
    audio_app = AudioStream()
    audio_app.animation()
    audio_app.saveResult()