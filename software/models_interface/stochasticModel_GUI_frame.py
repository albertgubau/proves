# GUI frame for the stochasticModel_function.py

from tkinter import *
import sys, os
from tkinter import filedialog, messagebox

import stochasticModel_function

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF

import matplotlib.pyplot as plt
import playsound
import essentia.standard as es
import numpy as np


class StochasticModel_frame:

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
        self.filelocation.insert(0, '../../sounds/ocean.wav')

        # BUTTON TO BROWSE SOUND FILE
        self.open_file = Button(self.parent, text="Browse...", command=self.browse_file)  # see: def browse_file(self)
        self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6))  # put it beside the filelocation textbox

        # BUTTON TO PREVIEW SOUND FILE
        self.preview = Button(self.parent, text=">", command=lambda: UF.wavplay(self.filelocation.get()))
        self.preview.grid(row=1, column=0, sticky=W, padx=(306, 6))

        ## STOCHASTIC MODEL

        # HOP SIZE
        H_label = "Hop size (H):"
        Label(self.parent, text=H_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10, 2))
        self.H = Entry(self.parent, justify=CENTER)
        self.H["width"] = 5
        self.H.grid(row=2, column=0, sticky=W, padx=(90, 5), pady=(10, 2))
        self.H.delete(0, END)
        self.H.insert(0, "256")

        # FFT size
        N_label = "FFT size (N):"
        Label(self.parent, text=N_label).grid(row=3, column=0, sticky=W, padx=5, pady=(10, 2))
        self.N = Entry(self.parent, justify=CENTER)
        self.N["width"] = 5
        self.N.grid(row=3, column=0, sticky=W, padx=(90, 5), pady=(10, 2))
        self.N.delete(0, END)
        self.N.insert(0, "512")

        # DECIMATION FACTOR
        stocf_label = "Decimation factor (bigger than 0, max of 1):"
        Label(self.parent, text=stocf_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10, 2))
        self.stocf = Entry(self.parent, justify=CENTER)
        self.stocf["width"] = 5
        self.stocf.grid(row=4, column=0, sticky=W, padx=(285, 5), pady=(10, 2))
        self.stocf.delete(0, END)
        self.stocf.insert(0, "0.1")

        # BUTTON TO COMPUTE EVERYTHING
        self.compute = Button(self.parent, text="Compute", command=self.compute_model)
        self.compute.grid(row=5, column=0, padx=5, pady=(10, 2), sticky=W)

        # BUTTON TO PLAY OUTPUT
        output_label = "Stochastic:"
        Label(self.parent, text=output_label).grid(row=6, column=0, sticky=W, padx=5, pady=(10, 15))
        self.output = Button(self.parent, text=">", command=lambda: UF.wavplay(
            'output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_stochasticModel.wav'))
        self.output.grid(row=6, column=0, padx=(80, 5), pady=(10, 15), sticky=W)

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
            inputFile = self.filelocation.get()

            fs = 44100  # Sampling rate

            H = int(self.H.get())
            N = int(self.N.get())
            stocf = float(self.stocf.get())

            # SMS-TOOLS version
            # stochasticModel_function.main(inputFile, H, N, stocf)

            ################################### ESSENTIA VERSION #######################################

            # create an audio loader and import audio file
            loader = es.MonoLoader(filename=inputFile, sampleRate=fs)
            x = loader()

            # output sound file (monophonic with sampling rate of 44100)
            outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stochasticModel.wav'

            # Algorithm instantiation
            stochasticAnal = es.StochasticModelAnal(fftSize=N, hopSize=H, stocf=stocf)
            stochasticSynth = es.StochasticModelSynth(sampleRate=fs, fftSize=N, hopSize=H, stocf=stocf)

            # writers
            awriter = es.MonoWriter(sampleRate=fs, filename=outputFile)

            # Frame counter
            frames = 0

            yst = np.array(0)  # initialize output array

            for frame in es.FrameGenerator(audio=x, frameSize=N, hopSize=H):

                # Analysis
                stocenv = stochasticAnal(frame)

                # Synthesis
                synth_yst = stochasticSynth(stocenv)

                # Save result
                yst = np.append(yst, synth_yst)

                if frames == 0:  # First frame
                    stocEnv = stocenv

                else:  # Next frames
                    stocEnv = np.vstack((stocEnv, stocenv))

                frames += 1

            awriter(yst)

            # frequency range to plot
            maxplotfreq = 15000.0

            # create figure to plot
            plt.figure(figsize=(9, 6))

            # plot the input sound
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(x.size) / float(fs), x)
            plt.axis([0, x.size / float(fs), min(x), max(x)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')
            plt.title('input sound: x')

            # plot stochastic representation   (Gave me problems)
            #plt.subplot(3, 1, 2)
            #numFrames = int(stocEnv[:, 0].size)
            #frmTime = H * np.arange(numFrames) / float(fs)
            #binFreq = np.arange(int(stocf * (N / 2 + 1))) * float(fs) / (stocf * N)
            #plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv), shading='auto')
            #plt.autoscale(tight=True)

            # plot spectrogram stochastic component
            plt.subplot(3, 1, 2)
            numFrames = int(stocEnv[:, 0].size)
            sizeEnv = int(stocEnv[0, :].size)
            frmTime = H * np.arange(numFrames) / float(fs)
            binFreq = (.5 * fs) * np.arange(sizeEnv * maxplotfreq / (.5 * fs)) / sizeEnv
            plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv[:, :int(sizeEnv * maxplotfreq / (.5 * fs) + 1)]))
            plt.autoscale(tight=True)
            plt.xlabel('time (sec)')
            plt.ylabel('frequency (Hz)')
            plt.title('stochastic approximation')

            # plot the output sound
            plt.subplot(3, 1, 3)
            plt.plot(np.arange(yst.size) / float(fs), yst)
            plt.axis([0, yst.size / float(fs), min(yst), max(yst)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')

            plt.tight_layout()
            plt.ion()
            plt.show()

            ############################################################################################

        except ValueError as errorMessage:
            messagebox.showerror("Input values error", errorMessage)
