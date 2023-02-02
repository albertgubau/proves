# GUI frame for the hprModel_function.py

# import required module
from scipy.fftpack import ifft, fftshift
from scipy.signal import get_window
import math
import numpy as np
from scipy.signal.windows import blackmanharris, triang, hann

from tkinter import *
import os
import sys
from tkinter import messagebox, filedialog

import matplotlib.pyplot as plt
import playsound
import essentia.standard as es
import hpsModel_function

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import hpsModel as HPS
import stochasticModel as STM
import sineModel as SM
import threading


class HpsModel_frame:

    def __init__(self, parent):

        self.parent = parent
        self.initUI()

    def initUI(self):

        choose_label = "Input file (.wav, mono and 44100 sampling rate):"
        Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10, 2))

        # TEXTBOX TO PRINT PATH OF THE SOUND FILE
        self.filelocation = Entry(self.parent)
        self.filelocation.focus_set()
        self.filelocation["width"] = 25
        self.filelocation.grid(row=1, column=0, sticky=W, padx=10)
        self.filelocation.delete(0, END)
        self.filelocation.insert(0, '../../sounds/sax-phrase-short.wav')

        # BUTTON TO BROWSE SOUND FILE
        self.open_file = Button(self.parent, text="Browse...", command=self.browse_file)  # see: def browse_file(self)
        self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6))  # put it beside the filelocation textbox

        # BUTTON TO PREVIEW SOUND FILE
        self.preview = Button(self.parent, text=">", command=lambda: playsound.playsound(self.filelocation.get()))
        self.preview.grid(row=1, column=0, sticky=W, padx=(306, 6))

        ## HARMONIC MODEL

        # ANALYSIS WINDOW TYPE
        wtype_label = "Window type:"
        Label(self.parent, text=wtype_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10, 2))
        self.w_type = StringVar()
        self.w_type.set("hann")  # initial value
        window_option = OptionMenu(self.parent, self.w_type, "hann", "hamming", "square",
                                   "blackmanharris92")  # What about blackman?
        window_option.grid(row=2, column=0, sticky=W, padx=(95, 5), pady=(10, 2))

        # WINDOW SIZE
        M_label = "Window size (M):"
        Label(self.parent, text=M_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10, 2))
        self.M = Entry(self.parent, justify=CENTER)
        self.M["width"] = 5
        self.M.grid(row=4, column=0, sticky=W, padx=(115, 5), pady=(10, 2))
        self.M.delete(0, END)
        self.M.insert(0, "601")

        # FFT SIZE
        N_label = "FFT size (N) (power of two bigger than M):"
        Label(self.parent, text=N_label).grid(row=5, column=0, sticky=W, padx=5, pady=(10, 2))
        self.N = Entry(self.parent, justify=CENTER)
        self.N["width"] = 5
        self.N.grid(row=5, column=0, sticky=W, padx=(270, 5), pady=(10, 2))
        self.N.delete(0, END)
        self.N.insert(0, "1024")

        # THRESHOLD MAGNITUDE
        t_label = "Magnitude threshold (t) (in dB):"
        Label(self.parent, text=t_label).grid(row=6, column=0, sticky=W, padx=5, pady=(10, 2))
        self.t = Entry(self.parent, justify=CENTER)
        self.t["width"] = 5
        self.t.grid(row=6, column=0, sticky=W, padx=(205, 5), pady=(10, 2))
        self.t.delete(0, END)
        self.t.insert(0, "-100")

        # MIN DURATION SINUSOIDAL TRACKS
        minSineDur_label = "Minimum duration of sinusoidal tracks:"
        Label(self.parent, text=minSineDur_label).grid(row=7, column=0, sticky=W, padx=5, pady=(10, 2))
        self.minSineDur = Entry(self.parent, justify=CENTER)
        self.minSineDur["width"] = 5
        self.minSineDur.grid(row=7, column=0, sticky=W, padx=(250, 5), pady=(10, 2))
        self.minSineDur.delete(0, END)
        self.minSineDur.insert(0, "0.1")

        # MAX NUMBER OF HARMONICS
        nH_label = "Maximum number of harmonics:"
        Label(self.parent, text=nH_label).grid(row=8, column=0, sticky=W, padx=5, pady=(10, 2))
        self.nH = Entry(self.parent, justify=CENTER)
        self.nH["width"] = 5
        self.nH.grid(row=8, column=0, sticky=W, padx=(215, 5), pady=(10, 2))
        self.nH.delete(0, END)
        self.nH.insert(0, "100")

        # MIN FUNDAMENTAL FREQUENCY
        minf0_label = "Minimum fundamental frequency:"
        Label(self.parent, text=minf0_label).grid(row=9, column=0, sticky=W, padx=5, pady=(10, 2))
        self.minf0 = Entry(self.parent, justify=CENTER)
        self.minf0["width"] = 5
        self.minf0.grid(row=9, column=0, sticky=W, padx=(220, 5), pady=(10, 2))
        self.minf0.delete(0, END)
        self.minf0.insert(0, "350")

        # MAX FUNDAMENTAL FREQUENCY
        maxf0_label = "Maximum fundamental frequency:"
        Label(self.parent, text=maxf0_label).grid(row=10, column=0, sticky=W, padx=5, pady=(10, 2))
        self.maxf0 = Entry(self.parent, justify=CENTER)
        self.maxf0["width"] = 5
        self.maxf0.grid(row=10, column=0, sticky=W, padx=(220, 5), pady=(10, 2))
        self.maxf0.delete(0, END)
        self.maxf0.insert(0, "700")

        # MAX ERROR ACCEPTED
        f0et_label = "Maximum error in f0 detection algorithm:"
        Label(self.parent, text=f0et_label).grid(row=11, column=0, sticky=W, padx=5, pady=(10, 2))
        self.f0et = Entry(self.parent, justify=CENTER)
        self.f0et["width"] = 5
        self.f0et.grid(row=11, column=0, sticky=W, padx=(265, 5), pady=(10, 2))
        self.f0et.delete(0, END)
        self.f0et.insert(0, "5")

        # ALLOWED DEVIATION OF HARMONIC TRACKS
        harmDevSlope_label = "Max frequency deviation in harmonic tracks:"
        Label(self.parent, text=harmDevSlope_label).grid(row=12, column=0, sticky=W, padx=5, pady=(10, 2))
        self.harmDevSlope = Entry(self.parent, justify=CENTER)
        self.harmDevSlope["width"] = 5
        self.harmDevSlope.grid(row=12, column=0, sticky=W, padx=(285, 5), pady=(10, 2))
        self.harmDevSlope.delete(0, END)
        self.harmDevSlope.insert(0, "0.01")

        # DECIMATION FACTOR
        stocf_label = "Stochastic approximation factor:"
        Label(self.parent, text=stocf_label).grid(row=13, column=0, sticky=W, padx=5, pady=(10, 2))
        self.stocf = Entry(self.parent, justify=CENTER)
        self.stocf["width"] = 5
        self.stocf.grid(row=13, column=0, sticky=W, padx=(210, 5), pady=(10, 2))
        self.stocf.delete(0, END)
        self.stocf.insert(0, "0.2")

        # BUTTON TO COMPUTE EVERYTHING
        self.compute = Button(self.parent, text="Compute", command=self.compute_model)
        self.compute.grid(row=14, column=0, padx=5, pady=(10, 2), sticky=W)

        # BUTTON TO PLAY SINE OUTPUT
        output_label = "Sinusoidal:"
        Label(self.parent, text=output_label).grid(row=15, column=0, sticky=W, padx=5, pady=(10, 0))
        self.output = Button(self.parent, text=">", command=lambda: UF.wavplay(
            'output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel_sines.wav'))
        self.output.grid(row=15, column=0, padx=(80, 5), pady=(10, 0), sticky=W)

        # BUTTON TO PLAY STOCHASTIC OUTPUT
        output_label = "Stochastic:"
        Label(self.parent, text=output_label).grid(row=16, column=0, sticky=W, padx=5, pady=(5, 0))
        self.output = Button(self.parent, text=">", command=lambda: UF.wavplay(
            'output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel_stochastic.wav'))
        self.output.grid(row=16, column=0, padx=(80, 5), pady=(5, 0), sticky=W)

        # BUTTON TO PLAY OUTPUT
        output_label = "Output:"
        Label(self.parent, text=output_label).grid(row=17, column=0, sticky=W, padx=5, pady=(5, 15))
        self.output = Button(self.parent, text=">", command=lambda: UF.wavplay(
            'output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel.wav'))
        self.output.grid(row=17, column=0, padx=(80, 5), pady=(5, 15), sticky=W)

        # define options for opening file
        self.file_opt = options = {}
        options['defaultextension'] = '.wav'
        options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
        options['initialdir'] = '../../sounds/'
        options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'

    def browse_file(self):

        self.filename = filedialog.askopenfilename(**self.file_opt)

        # set the text of the self.filelocation
        self.filelocation.delete(0, END)
        self.filelocation.insert(0, self.filename)

    def compute_model(self):

        try:

            # Parameters
            inputFile = self.filelocation.get()
            fs = 44100  # Sampling rate

            window = self.w_type.get()
            M = int(self.M.get())  # Window Size
            N = int(self.N.get())  # FFT size
            H = 128  # Hop size

            # What about those?
            minSineDur = float(self.minSineDur.get())
            f0et = int(self.f0et.get())  # is that freqDevOffset?

            stocf = float(self.stocf.get())

            hN = N // 2 + 1  # positive size of fft

            if (hN * stocf < 3):  # raise exception if decimation factor too small
                raise ValueError("Stochastic decimation factor too small")

            if (stocf > 1):  # raise exception if decimation factor too big
                raise ValueError("Stochastic decimation factor above 1")

            if (H <= 0):  # raise error if hop size 0 or negative
                raise ValueError("Hop size (H) smaller or equal to 0")

            if not (UF.isPower2(N)):  # raise error if N not a power of two
                raise ValueError("FFT size (N) is not a power of 2")

            ############################# ESSENTIA VERSION #########################################

            # create an audio loader and import audio file
            loader = es.MonoLoader(filename=inputFile, sampleRate=fs)
            x = loader()

            # Algorithm Instantation
            pitcher = es.PitchYin()

            hpsAnal = es.HpsModelAnal(fftSize=N,
                                      harmDevSlope=float(self.harmDevSlope.get()),
                                      magnitudeThreshold=int(self.t.get()),
                                      nHarmonics=int(self.nH.get()),
                                      minFrequency=int(self.minf0.get()),
                                      maxFrequency=int(self.maxf0.get()),
                                      stocf=stocf)

            stochasticAnal = es.StochasticModelAnal(fftSize=N, hopSize=H, stocf=stocf)

            sineSynth = es.SineModelSynth(sampleRate=fs, fftSize=N, hopSize=H)
            stochasticSynth = es.StochasticModelSynth(sampleRate=fs, fftSize=N, hopSize=H, stocf=stocf)
            ifft = es.IFFT(size=N)
            overl = es.OverlapAdd(frameSize=N, hopSize=H)




            # output sound file (monophonic with sampling rate of 44100)
            outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel_sines.wav'
            outputFileStochastic = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel_stochastic.wav'
            outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel.wav'

            # writers
            awriter = es.MonoWriter(sampleRate=fs, filename=outputFileSines)
            awriter2 = es.MonoWriter(sampleRate=fs, filename=outputFileStochastic)
            awriter3 = es.MonoWriter(sampleRate=fs, filename=outputFile)

            # Frame counter
            frames = 0

            yh = np.array(0)  # initialize output array
            yst = np.array(0)  # initialize output array

            for frame in es.FrameGenerator(audio=x, frameSize=N, hopSize=H):

                # Analysis (with Windowing)
                pitch = pitcher(frame)

                hps_anal = hpsAnal(frame, pitch[0])

                # Essentia already computes the stocEnv in the HpsModelAnal function, but I get bad results with that
                stocenv = stochasticAnal(frame)

                # Why in sms-tools uses Ns = 512 and H = 128?? It says that in synthesis uses that parameters, why?
                # Synthesis (with OverlapAdd and IFFT)
                fft_synth_yh = sineSynth(hps_anal[1], hps_anal[0], hps_anal[2])
                out_yh = overl(ifft(fft_synth_yh))
                # Save result
                yh = np.append(yh, out_yh)


                synth_yst = stochasticSynth(stocenv)

                # Here we do not apply the overlap add??

                # Save result
                yst = np.append(yst, synth_yst)

                if frames == 0:  # First frame
                    xhfreq = np.array([hps_anal[0]])
                    xhmag = np.array([hps_anal[1]])
                    xhphase = np.array([hps_anal[2]])
                    stocEnv = stocenv


                else:  # Next frames
                    xhfreq = np.vstack((xhfreq, np.array([hps_anal[0]])))
                    xhmag = np.vstack((xhmag, np.array([hps_anal[1]])))
                    xhphase = np.vstack((xhphase, np.array([hps_anal[2]])))
                    stocEnv = np.vstack((stocEnv, stocenv))

                frames += 1

            ########################################################################################

            # SMS-TOOLS Synthesis (Essentia doesn't have HpsModelSynth)
            # yh = SM.sineModelSynth(xhfreq, xhmag, xhphase, Ns, H, fs)  # synthesize harmonics
            # yst = STM.stochasticModelSynth(stocEnv, H, H * 2)  # synthesize stochastic residual
            y = yh[:min(yh.size, yst.size)] + yst[:min(yh.size, yst.size)]  # sum harmonic and stochastic components

            awriter(yh)
            awriter2(yst)
            awriter3(y)

            # create figure to plot
            plt.figure(figsize=(9, 6))

            # frequency range to plot
            maxplotfreq = 15000.0

            # plot the input sound
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(x.size) / float(fs), x)
            plt.axis([0, x.size / float(fs), min(x), max(x)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')
            plt.title('input sound: x')

            # plot spectrogram stochastic component
            plt.subplot(3, 1, 2)
            numFrames = int(stocEnv[:, 0].size)
            sizeEnv = int(stocEnv[0, :].size)
            frmTime = H * np.arange(numFrames) / float(fs)
            binFreq = (.5 * fs) * np.arange(sizeEnv * maxplotfreq / (.5 * fs)) / sizeEnv
            plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv[:, :int(sizeEnv * maxplotfreq / (.5 * fs) + 1)]))
            plt.autoscale(tight=True)

            # plot harmonic on top of stochastic spectrogram
            if (xhfreq.shape[1] > 0):
                harms = xhfreq * np.less(xhfreq, maxplotfreq)
                harms[harms == 0] = np.nan
                numFrames = harms.shape[0]
                frmTime = H * np.arange(numFrames) / float(fs)
                plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
                plt.xlabel('time (sec)')
                plt.ylabel('frequency (Hz)')
                plt.autoscale(tight=True)
                plt.title('harmonics + stochastic spectrogram')

            # plot the output sound
            plt.subplot(3, 1, 3)
            plt.plot(np.arange(y.size) / float(fs), y)
            plt.axis([0, y.size / float(fs), min(y), max(y)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')
            plt.title('output sound: y')

            plt.tight_layout()
            plt.ion()
            plt.show()

        except ValueError as errorMessage:
            messagebox.showerror("Input values error", str(errorMessage))
