# GUI frame for the hprModel_function.py

from tkinter import *
import sys, os
from tkinter import filedialog, messagebox

import hprModel_function
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF

import numpy as np
import matplotlib.pyplot as plt
import playsound
import essentia.standard as es
 
class HprModel_frame:
  
    def __init__(self, parent):

        self.parent = parent
        self.initUI()

    def initUI(self):

        choose_label = "Input file (.wav, mono and 44100 sampling rate):"
        Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
        #TEXTBOX TO PRINT PATH OF THE SOUND FILE
        self.filelocation = Entry(self.parent)
        self.filelocation.focus_set()
        self.filelocation["width"] = 25
        self.filelocation.grid(row=1,column=0, sticky=W, padx=10)
        self.filelocation.delete(0, END)
        self.filelocation.insert(0, '../../sounds/sax-phrase-short.wav')

        #BUTTON TO BROWSE SOUND FILE
        self.open_file = Button(self.parent, text="Browse...", command=self.browse_file) #see: def browse_file(self)
        self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
        #BUTTON TO PREVIEW SOUND FILE
        self.preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()))
        self.preview.grid(row=1, column=0, sticky=W, padx=(306,6))

        ## HARMONIC MODEL

        #ANALYSIS WINDOW TYPE
        wtype_label = "Window type:"
        Label(self.parent, text=wtype_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10,2))
        self.w_type = StringVar()
        self.w_type.set("blackman") # initial value
        window_option = OptionMenu(self.parent, self.w_type, "rectangular", "hann", "hamming", "blackman", "blackmanharris")
        window_option.grid(row=2, column=0, sticky=W, padx=(95,5), pady=(10,2))

        #WINDOW SIZE
        M_label = "Window size (M):"
        Label(self.parent, text=M_label).grid(row=3, column=0, sticky=W, padx=5, pady=(10,2))
        self.M = Entry(self.parent, justify=CENTER)
        self.M["width"] = 5
        self.M.grid(row=3,column=0, sticky=W, padx=(115,5), pady=(10,2))
        self.M.delete(0, END)
        self.M.insert(0, "601")

        #FFT SIZE
        N_label = "FFT size (N) (power of two bigger than M):"
        Label(self.parent, text=N_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10,2))
        self.N = Entry(self.parent, justify=CENTER)
        self.N["width"] = 5
        self.N.grid(row=4,column=0, sticky=W, padx=(270,5), pady=(10,2))
        self.N.delete(0, END)
        self.N.insert(0, "1024")

        #THRESHOLD MAGNITUDE
        t_label = "Magnitude threshold (t) (in dB):"
        Label(self.parent, text=t_label).grid(row=5, column=0, sticky=W, padx=5, pady=(10,2))
        self.t = Entry(self.parent, justify=CENTER)
        self.t["width"] = 5
        self.t.grid(row=5, column=0, sticky=W, padx=(205,5), pady=(10,2))
        self.t.delete(0, END)
        self.t.insert(0, "-100")

        #MIN DURATION SINUSOIDAL TRACKS
        minSineDur_label = "Minimum duration of harmonic tracks:"
        Label(self.parent, text=minSineDur_label).grid(row=6, column=0, sticky=W, padx=5, pady=(10,2))
        self.minSineDur = Entry(self.parent, justify=CENTER)
        self.minSineDur["width"] = 5
        self.minSineDur.grid(row=6, column=0, sticky=W, padx=(250,5), pady=(10,2))
        self.minSineDur.delete(0, END)
        self.minSineDur.insert(0, "0.1")

        #MAX NUMBER OF HARMONICS
        nH_label = "Maximum number of harmonics:"
        Label(self.parent, text=nH_label).grid(row=7, column=0, sticky=W, padx=5, pady=(10,2))
        self.nH = Entry(self.parent, justify=CENTER)
        self.nH["width"] = 5
        self.nH.grid(row=7, column=0, sticky=W, padx=(215,5), pady=(10,2))
        self.nH.delete(0, END)
        self.nH.insert(0, "100")

        #MIN FUNDAMENTAL FREQUENCY
        minf0_label = "Minimum fundamental frequency:"
        Label(self.parent, text=minf0_label).grid(row=8, column=0, sticky=W, padx=5, pady=(10,2))
        self.minf0 = Entry(self.parent, justify=CENTER)
        self.minf0["width"] = 5
        self.minf0.grid(row=8, column=0, sticky=W, padx=(220,5), pady=(10,2))
        self.minf0.delete(0, END)
        self.minf0.insert(0, "350")

        #MAX FUNDAMENTAL FREQUENCY
        maxf0_label = "Maximum fundamental frequency:"
        Label(self.parent, text=maxf0_label).grid(row=9, column=0, sticky=W, padx=5, pady=(10,2))
        self.maxf0 = Entry(self.parent, justify=CENTER)
        self.maxf0["width"] = 5
        self.maxf0.grid(row=9, column=0, sticky=W, padx=(220,5), pady=(10,2))
        self.maxf0.delete(0, END)
        self.maxf0.insert(0, "700")

        #MAX ERROR ACCEPTED
        f0et_label = "Maximum error in f0 detection algorithm:"
        Label(self.parent, text=f0et_label).grid(row=10, column=0, sticky=W, padx=5, pady=(10,2))
        self.f0et = Entry(self.parent, justify=CENTER)
        self.f0et["width"] = 5
        self.f0et.grid(row=10, column=0, sticky=W, padx=(265,5), pady=(10,2))
        self.f0et.delete(0, END)
        self.f0et.insert(0, "5")

        #ALLOWED DEVIATION OF HARMONIC TRACKS
        harmDevSlope_label = "Max frequency deviation in harmonic tracks:"
        Label(self.parent, text=harmDevSlope_label).grid(row=11, column=0, sticky=W, padx=5, pady=(10,2))
        self.harmDevSlope = Entry(self.parent, justify=CENTER)
        self.harmDevSlope["width"] = 5
        self.harmDevSlope.grid(row=11, column=0, sticky=W, padx=(285,5), pady=(10,2))
        self.harmDevSlope.delete(0, END)
        self.harmDevSlope.insert(0, "0.01")

        #BUTTON TO COMPUTE EVERYTHING
        self.compute = Button(self.parent, text="Compute", command=self.compute_model)
        self.compute.grid(row=12, column=0, padx=5, pady=(10,2), sticky=W)

        #BUTTON TO PLAY SINE OUTPUT
        output_label = "Sinusoidal:"
        Label(self.parent, text=output_label).grid(row=13, column=0, sticky=W, padx=5, pady=(10,0))
        self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hprModel_sines.wav'))
        self.output.grid(row=13, column=0, padx=(80,5), pady=(10,0), sticky=W)

        #BUTTON TO PLAY RESIDUAL OUTPUT
        output_label = "Residual:"
        Label(self.parent, text=output_label).grid(row=14, column=0, sticky=W, padx=5, pady=(5,0))
        self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hprModel_residual.wav'))
        self.output.grid(row=14, column=0, padx=(80,5), pady=(5,0), sticky=W)

        #BUTTON TO PLAY OUTPUT
        output_label = "Output:"
        Label(self.parent, text=output_label).grid(row=15, column=0, sticky=W, padx=5, pady=(5,15))
        self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hprModel.wav'))
        self.output.grid(row=15, column=0, padx=(80,5), pady=(5,15), sticky=W)


        # define options for opening file
        self.file_opt = options = {}
        options['defaultextension'] = '.wav'
        options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
        options['initialdir'] = '../../sounds/'
        options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'
 
    def browse_file(self):

        self.filename = filedialog.askopenfilename(**self.file_opt)
 
        #set the text of the self.filelocation
        self.filelocation.delete(0, END)
        self.filelocation.insert(0,self.filename)

    def compute_model(self):

        try:
            inputFile = self.filelocation.get()
            fs = 44100

            window = self.w_type.get()
            M = int(self.M.get())
            N = int(self.N.get())

            H = N//4

            t = int(self.t.get())
            minSineDur = float(self.minSineDur.get())
            nH = int(self.nH.get())
            minf0 = int(self.minf0.get())
            maxf0 = int(self.maxf0.get())
            f0et = int(self.f0et.get())
            harmDevSlope = float(self.harmDevSlope.get())

            #hprModel_function.main(inputFile, window, M, N, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope)

            ############################# ESSENTIA VERSION #########################################

            # create an audio loader and import audio file
            loader = es.MonoLoader(filename=inputFile, sampleRate=fs)
            x = loader()

            # Algorithm Instantation
            pitcher = es.PitchYin()

            hprAnal = es.HprModelAnal(fftSize=N,
                                      harmDevSlope=harmDevSlope,
                                      magnitudeThreshold=t,
                                      nHarmonics=nH,
                                      minFrequency=minf0,
                                      maxFrequency=maxf0)

            sineSubtraction = es.SineSubtraction(fftSize=N,
                                                 hopSize=H,
                                                 sampleRate=fs)

            sineSynth = es.SineModelSynth(sampleRate=fs, fftSize=N, hopSize=H)


            ifft = es.IFFT(size=N)
            overl = es.OverlapAdd(frameSize=N, hopSize=H)

            # output sound file (monophonic with sampling rate of 44100)
            outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_sines.wav'
            outputFileResidual = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_residual.wav'
            outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel.wav'

            # writers
            awriter = es.MonoWriter(sampleRate=fs, filename=outputFileSines)
            awriter2 = es.MonoWriter(sampleRate=fs, filename=outputFileResidual)
            awriter3 = es.MonoWriter(sampleRate=fs, filename=outputFile)

            # Frame counter
            frames = 0

            yh = np.array(0)  # initialize output array
            yr = np.array(0)  # initialize output array

            for frame in es.FrameGenerator(audio=x, frameSize=N, hopSize=H):

                # Analysis (with Windowing)
                pitch = pitcher(frame)

                hpr_anal = hprAnal(frame, pitch[0])

                # Essentia already computes the residual in the SprModelAnal function, but I get bad results with that
                xr = sineSubtraction(frame, hpr_anal[1], hpr_anal[0], hpr_anal[2])


                # Synthesis (with OverlapAdd and IFFT)
                fft_synth_yh = sineSynth(hpr_anal[1], hpr_anal[0], hpr_anal[2])
                out_yh = overl(ifft(fft_synth_yh))

                # Save result
                yh = np.append(yh, out_yh)
                yr = np.append(yr, xr)

                if frames == 0:  # First frame
                    hfreq = np.array([hpr_anal[0]])
                    xhmag = np.array([hpr_anal[1]])
                    xhphase = np.array([hpr_anal[2]])

                else:  # Next frames
                    hfreq = np.vstack((hfreq, np.array([hpr_anal[0]])))
                    xhmag = np.vstack((xhmag, np.array([hpr_anal[1]])))
                    xhphase = np.vstack((xhphase, np.array([hpr_anal[2]])))

                frames += 1

            ########################################################################################

            # SMS-TOOLS Synthesis (Essentia doesn't have HpsModelSynth)
            # yh = SM.sineModelSynth(xhfreq, xhmag, xhphase, Ns, H, fs)  # synthesize harmonics
            # yst = STM.stochasticModelSynth(stocEnv, H, H * 2)  # synthesize stochastic residual
            y = yh[:min(yh.size, yr.size)]+yr[:min(yh.size, yr.size)]  # sum harmonic and residual components

            awriter(yh)
            awriter2(yr)
            awriter3(y)

            # create figure to plot
            plt.figure(figsize=(9, 6))

            # frequency range to plot
            maxplotfreq = 5000.0

            # plot the input sound
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(x.size) / float(fs), x)
            plt.axis([0, x.size / float(fs), min(x), max(x)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')
            plt.title('input sound: x')

            # plot the magnitude spectrogram of residual
            plt.subplot(3, 1, 2)
            #maxplotbin = int(N * maxplotfreq / fs)
            #numFrames = int(mXr[:, 0].size)
            #frmTime = H * np.arange(numFrames) / float(fs)
            #binFreq = np.arange(maxplotbin + 1) * float(fs) / N
            #plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:, :maxplotbin + 1]))
            #plt.autoscale(tight=True)

            # plot harmonic frequencies on residual spectrogram
            if (hfreq.shape[1] > 0):
                harms = hfreq * np.less(hfreq, maxplotfreq)
                harms[harms == 0] = np.nan
                numFrames = int(harms[:, 0].size)
                frmTime = H * np.arange(numFrames) / float(fs)
                plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
                plt.xlabel('time(s)')
                plt.ylabel('frequency(Hz)')
                plt.autoscale(tight=True)
                plt.title('harmonics + residual spectrogram')

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
