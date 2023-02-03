# GUI frame for the sineModel_function.py

from tkinter import *
import sys, os
from tkinter import messagebox, filedialog

import sineModel_function

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import sineModel as SM



import essentia.standard as es
# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')


class SineModel_frame:

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
        self.filelocation.insert(0, '../../sounds/bendir.wav')

        # BUTTON TO BROWSE SOUND FILE
        self.open_file = Button(self.parent, text="Browse...", command=self.browse_file)  # see: def browse_file(self)
        self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6))  # put it beside the filelocation textbox

        # BUTTON TO PREVIEW SOUND FILE
        self.preview = Button(self.parent, text=">", command=lambda: UF.wavplay(self.filelocation.get()))
        self.preview.grid(row=1, column=0, sticky=W, padx=(306, 6))

        ## SINE MODEL

        # ANALYSIS WINDOW TYPE
        wtype_label = "Window type:"
        Label(self.parent, text=wtype_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10, 2))
        self.w_type = StringVar()
        self.w_type.set("hamming")  # initial value
        window_option = OptionMenu(self.parent, self.w_type, "square", "hann", "hamming","triangular",
                                   "blackmanharris62","blackmanharris70", "blackmanharris74", "blackmanharris92")
        window_option.grid(row=2, column=0, sticky=W, padx=(95, 5), pady=(10, 2))

        # WINDOW SIZE
        M_label = "Window size (M):"
        Label(self.parent, text=M_label).grid(row=3, column=0, sticky=W, padx=5, pady=(10, 2))
        self.M = Entry(self.parent, justify=CENTER)
        self.M["width"] = 5
        self.M.grid(row=3, column=0, sticky=W, padx=(115, 5), pady=(10, 2))
        self.M.delete(0, END)
        self.M.insert(0, "2001")

        # FFT SIZE
        N_label = "FFT size (N) (power of two bigger than M):"
        Label(self.parent, text=N_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10, 2))
        self.N = Entry(self.parent, justify=CENTER)
        self.N["width"] = 5
        self.N.grid(row=4, column=0, sticky=W, padx=(270, 5), pady=(10, 2))
        self.N.delete(0, END)
        self.N.insert(0, "2048")

        # THRESHOLD MAGNITUDE
        t_label = "Magnitude threshold (t) (in dB):"
        Label(self.parent, text=t_label).grid(row=5, column=0, sticky=W, padx=5, pady=(10, 2))
        self.t = Entry(self.parent, justify=CENTER)
        self.t["width"] = 5
        self.t.grid(row=5, column=0, sticky=W, padx=(205, 5), pady=(10, 2))
        self.t.delete(0, END)
        self.t.insert(0, "-80")

        # MIN DURATION SINUSOIDAL TRACKS
        minSineDur_label = "Minimum duration of sinusoidal tracks:"
        Label(self.parent, text=minSineDur_label).grid(row=6, column=0, sticky=W, padx=5, pady=(10, 2))
        self.minSineDur = Entry(self.parent, justify=CENTER)
        self.minSineDur["width"] = 5
        self.minSineDur.grid(row=6, column=0, sticky=W, padx=(250, 5), pady=(10, 2))
        self.minSineDur.delete(0, END)
        self.minSineDur.insert(0, "0.02")

        # MAX NUMBER PARALLEL SINUSOIDS
        maxnSines_label = "Maximum number of parallel sinusoids:"
        Label(self.parent, text=maxnSines_label).grid(row=7, column=0, sticky=W, padx=5, pady=(10, 2))
        self.maxnSines = Entry(self.parent, justify=CENTER)
        self.maxnSines["width"] = 5
        self.maxnSines.grid(row=7, column=0, sticky=W, padx=(250, 5), pady=(10, 2))
        self.maxnSines.delete(0, END)
        self.maxnSines.insert(0, "150")

        # FREQUENCY DEVIATION ALLOWED
        freqDevOffset_label = "Max frequency deviation in sinusoidal tracks (at freq 0):"
        Label(self.parent, text=freqDevOffset_label).grid(row=8, column=0, sticky=W, padx=5, pady=(10, 2))
        self.freqDevOffset = Entry(self.parent, justify=CENTER)
        self.freqDevOffset["width"] = 5
        self.freqDevOffset.grid(row=8, column=0, sticky=W, padx=(350, 5), pady=(10, 2))
        self.freqDevOffset.delete(0, END)
        self.freqDevOffset.insert(0, "10")

        # SLOPE OF THE FREQ DEVIATION
        freqDevSlope_label = "Slope of the frequency deviation (as function of freq):"
        Label(self.parent, text=freqDevSlope_label).grid(row=9, column=0, sticky=W, padx=5, pady=(10, 2))
        self.freqDevSlope = Entry(self.parent, justify=CENTER)
        self.freqDevSlope["width"] = 5
        self.freqDevSlope.grid(row=9, column=0, sticky=W, padx=(340, 5), pady=(10, 2))
        self.freqDevSlope.delete(0, END)
        self.freqDevSlope.insert(0, "0.001")

        # BUTTON TO COMPUTE EVERYTHING
        self.compute = Button(self.parent, text="Compute", command=self.compute_model)
        self.compute.grid(row=10, column=0, padx=5, pady=(10, 2), sticky=W)

        # BUTTON TO PLAY OUTPUT
        output_label = "Output:"
        Label(self.parent, text=output_label).grid(row=11, column=0, sticky=W, padx=5, pady=(10, 15))
        self.output = Button(self.parent, text=">", command=lambda: UF.wavplay(
            'output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_sineModel.wav'))
        self.output.grid(row=11, column=0, padx=(60, 5), pady=(10, 15), sticky=W)

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
            window = self.w_type.get()
            M = int(self.M.get())
            N = int(self.N.get())
            t = int(self.t.get())
            minSineDur = float(self.minSineDur.get())
            maxnSines = int(self.maxnSines.get())
            freqDevOffset = int(self.freqDevOffset.get())
            freqDevSlope = float(self.freqDevSlope.get())

            # SMS-TOOLS version
            #sineModel_function.main(inputFile, window, M, N, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope)


            ############################# ESSENTIA VERSION #########################################

            fs = 44100

            # hop size (has to be 1/4 of N)
            H = N // 4

            # create an audio loader and import audio file
            loader = es.MonoLoader(filename=inputFile, sampleRate=fs)
            x = loader()

            # Algorithm Instantation
            w = es.Windowing(type=window, size=M - 1)  # Check if the window size has some effect on the result
            fft = es.FFT(size=N)

            sineAnal = es.SineModelAnal(sampleRate=fs,
                                        maxnSines=maxnSines,
                                        magnitudeThreshold=t,
                                        freqDevOffset=freqDevOffset,
                                        freqDevSlope=freqDevSlope)

            sineSynth = es.SineModelSynth(sampleRate=fs,
                                          fftSize=N,
                                          hopSize=H)
            ifft = es.IFFT(size=N)
            overl = es.OverlapAdd(frameSize=N, hopSize=H)

            # Output sound file location
            outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModel.wav'

            # Writer
            awrite = es.MonoWriter(filename=outputFile, sampleRate=fs)

            y = np.array([])  # initialize output array

            frames = 0

            for frame in es.FrameGenerator(audio=x, frameSize=N, hopSize=H):

                # Analysis (with FFT and Windowing)
                infft = fft(w(frame))


                sine_anal = sineAnal(infft)

                # Synthesis (with OverlapAdd and IFFT)
                fft_synth = sineSynth(sine_anal[1], sine_anal[0], sine_anal[2])

                if frames == 0:
                    tfreq = np.array([sine_anal[0]])

                else:
                    tfreq = np.vstack((tfreq, np.array([sine_anal[0]])))

                out = overl(ifft(fft_synth))

                # Save result
                y = np.append(y, out)

                frames += 1

            # Write the output file to the specified location
            awrite(y)

            mplstyle.use('fast')
            mpl.rcParams['path.simplify'] = True
            mpl.rcParams['path.simplify_threshold'] = 1.0

            plt.figure(figsize=(16, 9))
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(x.size) / float(fs), x)
            plt.axis([0, x.size / float(fs), min(x), max(x)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')
            plt.title('input sound: x')

            # This plot is not correct I think, maybe for the result of applying the essentia function
            plt.subplot(3, 1, 2)
            if (tfreq.shape[1] > 0):
                numFrames = tfreq.shape[0]
                frmTime = H * np.arange(numFrames) / float(fs)
                tfreq[tfreq <= 0] = np.nan
                plt.plot(frmTime, tfreq)
                plt.axis([0, x.size / float(fs), 0, 5000.0])
                plt.title('frequencies of sinusoidal tracks')

            plt.subplot(3, 1, 3)
            plt.plot(np.arange(y.size) / float(fs), y)
            plt.axis([0, y.size / float(fs), min(y), max(y)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')
            plt.title('output sound: y')

            plt.tight_layout()
            plt.ion()
            plt.show()

            ########################################################################################


        except ValueError as errorMessage:
            messagebox.showerror("Input values error", str(errorMessage))
