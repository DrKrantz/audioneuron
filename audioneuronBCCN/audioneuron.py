import numpy as np
import pyaudio
import wave
import sys
#import pylab as pl
import matplotlib.collections as collections
import time
from threading import Thread
from io import StringIO

import pygame

#import Tkinter
#import FileDialog
import valuehandler
from neuroncontrol import DestexheNeuron
from settings import *
from pygamedisplay import FullDisplay
#from visualizer import FFTPlotter


p = pyaudio.PyAudio()

valueHandler = valuehandler.ValueHandler()

class SoundHandler:
    FORMAT = pyaudio.paInt16
    RATE = 44100
    CHANNELS = 2
    CHUNK = 1024
    def __init__(self):
        self.__createSound()
        self.__createStream()
        
    def __createStream(self):
        self.__stream = p.open(format = self.FORMAT,
                channels = self.CHANNELS,
                rate = self.RATE,
                input = True,
                output = True,
                frames_per_buffer = self.CHUNK)
    
    def __createSound(self):
        numSamples = int(self.RATE*toneDuration/1000.)
        volumeScale = (.85)*32767
        twopi = 2*np.pi
        sine = np.sin( np.arange(numSamples)*twopi*neuronalFrequency/self.RATE )*volumeScale
        
        if channel=='R':
            signal = np.array((np.zeros_like(sine),sine))
        elif channel=='L':
            signal = np.array((sine,np.zeros_like(sine)))
        elif channel=='LR':
            signal = np.array((sine,sine))
        else:
            print(('ERROR: unknown channel'. channel))
        signal = signal.transpose().flatten()
          
        # add an- u- abschwellen
        if dampDuration<toneDuration/2.:
            numDampSamp = 2*int(self.RATE*dampDuration) # the 2 is the stereo
        else:
            print('WARNING! dampDuration>toneDuration/2!!! Using toneDuration/2')
            numDampSamp = 2*int(self.RATE*toneDuration/2.) # the 2 is the stereo
        dampVec = np.linspace(0,1,numDampSamp)
        signal[0:numDampSamp] *= dampVec
        signal[len(signal)-numDampSamp::] *= dampVec[::-1]
        signal = np.append(signal,np.zeros(2*int(self.RATE*endSilence)))
    #        import pylab as pl
    #        pl.plot(signal)
    #        pl.show()
        self.__newdata = signal.astype(np.int16).tostring()
    
    def play(self):
        idx = 0
        data = self.__newdata[idx:idx+4*self.CHUNK]
        while data != '':
            self.__stream.write(data)
            idx += 4*self.CHUNK
            data = self.__newdata[idx:idx+4*self.CHUNK]
            
class OutputHandler:
    def __init__(self,intervals):
        self.__display = FullDisplay(playedFrequency = neuronalFrequency,
                                     frequencies=presynapticFrequencies,
                    intervals = intervals,
                    types=list(presynapticNeurons.values()),
                    width=displaySize[0],height=displaySize[1])
        
        self.__player = SoundHandler()
            
    def update(self):
        self.__display.update()
        if valueHandler['hasSpiked']:
            self.__player.play()
    
    def forcePlay(self):
        self.__player.play()
    
    def toggleFullscreen(self):
        self.__display.toggleFullscreen()

class ThreadRecorder(Thread):
    SWIDTH = pyaudio.get_sample_size(SoundHandler.FORMAT)
    def __init__(self,inputName='Built-in Microphone',channelId=1,refreshInterval = 0.05):
        super(ThreadRecorder,self).__init__()
        self.__refreshInterval = refreshInterval #s
        self.__nread = int(self.__refreshInterval*SoundHandler.RATE)
        self.__p = pyaudio.PyAudio()
        self.__createStream(inputName,channelId)
        self.__isRecording = True
        
    def __createStream(self,inputName,channelId):
        index = self.__getIndexByName(inputName)
        if index>=0:
            self.__stream = self.__p.open(format = SoundHandler.FORMAT,
                channels = channelId,
                rate = SoundHandler.RATE,
                input = True,
                input_device_index = index,
                frames_per_buffer = SoundHandler.CHUNK)
            print(('SETUP input:', inputName, 'connected'))
        else:
            ''' SEND TO STOUT? '''
            print(('no such input device', inputName))
        
    def __getIndexByName(self,inputName):
        n = self.__p.get_device_count()
        index = -1
        for k in range(n):
            inf = self.__p.get_device_info_by_index(k)
            if inf['name']==inputName and inf['maxInputChannels']>0:
                index = inf['index']
        return index
    
    def setEngineCb(self,engineCb):
        self.__sendToEngine = engineCb 
    
    def run(self): # has to be named 'run', because Thread.start() calls 'run'!!!
        while self.__isRecording:
            nbits = self.__stream.get_read_available()
            realdata = None
            try:
                data = self.__stream.read(self.__nread)
                realdata = np.array(wave.struct.unpack("%dh"%(len(data)/self.SWIDTH),data))
            except IOError as ex:
                pass
            if realdata is not None:
                self.__sendToEngine(realdata)
                
    def stop(self):
        self.__isRecording = False        

class Recorder:
    SWIDTH = pyaudio.get_sample_size(SoundHandler.FORMAT)
    def __init__(self,inputName='Microphone',channelId=1,refreshInterval = 0.05):
        self.__refreshInterval = refreshInterval #s
        self.__nread = int(self.__refreshInterval*SoundHandler.RATE)
        self.__createStream(inputName,channelId)
        self.__sendToEngine = None
        
    def setEngineCb(self,engineCb):
        self.__sendToEngine = engineCb 
        
    def __createStream(self,inputName,channelId):
        self.__stream = p.open(format = SoundHandler.FORMAT,
            channels = channelId,
            rate = SoundHandler.RATE,
            input = True,
            input_device_index = self.__getIndexByName(inputName),
            frames_per_buffer = SoundHandler.CHUNK)
        
    def __getIndexByName(self,inputName):
        return 0
    
    def record(self):
        nbits = self.__stream.get_read_available()
        try:
            data = self.__stream.read(self.__nread)
            realdata = np.array(wave.struct.unpack("%dh"%(len(data)/self.SWIDTH),data))
            self.__sendToEngine(realdata)
        except IOError as ex:
            print(('skipping audio', nbits))
            data = self.__stream.read(self.__nread)
            realdata = np.array(wave.struct.unpack("%dh"%(len(data)/self.SWIDTH),data))
            self.__sendToEngine(realdata)

class FrequencyDetector:
    def __init__(self,frequencies=None,threshold=150, tolerance=.1):
        self.__frequencies = frequencies 
        self.__frequencyIntervals = []
        self.__createFrequencyIntervals(frequencies,tolerance)
        self.__threshold = threshold
        self.__timeDetected = np.zeros_like(frequencies)
        
    @property
    def intervals(self):
        return self.__frequencyIntervals

    def get(self,xData,fftData):
        return self.__detect(xData,fftData)
    
    def __createFrequencyIntervals(self,freqs,tol):
        lower = (1-tol*(1-1/NOTERATIO))
        upper = (1+tol*(NOTERATIO-1))
        [self.__frequencyIntervals.append([freq*lower,freq*upper]) for freq in freqs]
    
    def __detect(self,xData,fftData):
        actList = len(self.__frequencies)*[True]
        for id,(mn,mx) in enumerate(self.__frequencyIntervals):
            if time.time() - self.__timeDetected[id] > toneDuration/1000.: 
                idx, = np.nonzero((xData>=mn) & (xData<=mx)) 
                volume = np.mean(fftData[idx])
                if volume>self.__threshold:
                    self.__timeDetected[id] = time.time()
                else:
                    actList[id] = False
            else:
                actList[id] = False
        if np.any(actList):
            idx, = np.nonzero(actList)
            print(('detected:', np.array(self.__frequencies)[idx]))
        return actList
    
class FrequencyIdentifyer:
    def __init__(self,minFreq=1e2,maxFreq=1e4):
        self.__minFreq=minFreq
        self.__maxFreq=maxFreq

    def get(self,xData,fftData):
        return self.__detect(xData,fftData)
    
    def __detect(self,xData,fftData):
        return xData[np.argmax(fftData)]
    

    
class InputEngine:
    def __init__(self,recorder=Recorder(refreshInterval = 0.1)):
        self.attach(1)
        self.__plot = plot
        self.__detector = FrequencyDetector(frequencies = presynapticFrequencies,
                                            tolerance = frequencyTolerance,
                                            threshold = frequencyThreshold)
        self.__recorder = recorder
        self.__recorder.setEngineCb(self.__onAudioReceive)
        
    @property
    def intervals(self):
        return self.__detector.intervals
    
    def setOutputCb(self,pyfunc):
        self.__outputCb = pyfunc
        
    def update(self):
        self.__recorder.record()
        
    def attach(self,neuronId):
        '''connect a neuron to this Microphone, adding the freqs and the update Callback'''
        self.__neuron = DestexheNeuron()
        pars = defaultPars(neuronalType)
        pars.update(neuronParameters)
        valueHandler.update(**pars)
        self.__neuron.setParams(presynapticNeurons = presynapticNeurons,**pars)
        
    def __onAudioReceive(self,data):
        if len(data)>1:
            fftData=np.fft.fft(data)/data.size
            fftData = np.abs(fftData[list(range(data.size/2))])
            frqs = np.arange(data.size)/(data.size/float(SoundHandler.RATE))
            xData = frqs[list(range(data.size/2))]
            valueHandler.update(xData=xData,fftData=fftData)
            detectedFreqs  = self.__detector.get(xData,fftData)
            valueHandler.update(detectedFreqs=detectedFreqs)
            vals = self.__neuron.update()

            ''' SEND NEXT LINE TO A SUBPROCESS / MULTIPROCESS ! '''
            self.__outputCb()#(xData,fftData,self.__neuron._v,vals)

class MainApp:
    def __init__(self,plot=False):
        pygame.init()
        self.__inputEngine = InputEngine()
        self.__outputHandler = OutputHandler(self.__inputEngine.intervals)
        self.__inputEngine.setOutputCb(self.__outputHandler.update)
        self.__fullscreen = False
        
    def input(self,events):
        for event in events: 
            if event.type == pygame.locals.QUIT:
                sys.exit(0)
            elif event.type == pygame.locals.MOUSEBUTTONDOWN:
                self.__outputHandler.forcePlay()
            elif event.type == pygame.locals.KEYDOWN:
                if event.dict['key'] == pygame.locals.K_f:
                    self.__fullscreen = not(self.__fullscreen)
                    if self.__fullscreen:
                        pygame.display.set_mode(displaySize,pygame.locals.FULLSCREEN)
                    else:
                        pygame.display.set_mode(displaySize)
    
    def run(self):
        updInt = .05
        now = time.time()
        while True:
            self.input(pygame.event.get())
            if time.time()-now<updInt:
                pass
            else:
                self.__inputEngine.update()
                now = time.time()
    
if __name__=='__main__':
    
    plot = True
    if len(sys.argv)>1:
        print((sys.argv))
        if sys.argv[-1] == 'p':
            plot = True
    app = MainApp(plot)
    app.run()
    
#    input = InputEngine(plot=plot)
#    raw_input('los?')
    