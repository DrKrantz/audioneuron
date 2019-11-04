import numpy as np
import pyaudio
import wave
import sys
import time
from threading import Thread

import pygame

import valuehandler
from neuroncontrol import DestexheNeuron
import settings
from pygamedisplay import FullDisplay


p = pyaudio.PyAudio()
valueHandler = valuehandler.ValueHandler()


class SoundPlayer:
    RATE = 44100
    CHANNELS = 2

    def __init__(self):
        self.__sound = np.array([])
        self.__create_sound()
    
    def __create_sound(self):
        num_samples = int(self.RATE*settings.toneDuration/1000.)
        bits = 16
        twopi = 2*np.pi

        pygame.mixer.pre_init(44100, -bits, self.CHANNELS)
        pygame.init()

        max_sample = 2 ** (bits - 1) - 1
        sine = max_sample * np.sin(np.arange(num_samples)*twopi*settings.neuronalFrequency/self.RATE)

        #  add an- u- abschwellen
        if settings.dampDuration < settings.toneDuration / 2.:
            num_damp_samp = int(self.RATE * settings.dampDuration)
        else:
            print('WARNING! dampDuration>toneDuration/2!!! Using toneDuration/2')
            num_damp_samp = int(self.RATE * settings.toneDuration / 2.)  # the 2 is the stereo
        damp_vec = np.linspace(0, 1, num_damp_samp)
        sine[0:num_damp_samp] *= damp_vec
        sine[len(sine) - num_damp_samp::] *= damp_vec[::-1]

        #  add ending silence
        sine = np.concatenate([sine, np.zeros(int(self.RATE * settings.endSilence))])

        #  generate signal for channels
        signal = np.array((sine, sine))  # default to LR
        if settings.channel == 'R':
            signal = np.array((np.zeros_like(sine), sine))
        elif settings.channel == 'L':
            signal = np.array((sine, np.zeros_like(sine)))
        elif settings.channel == 'LR':
            pass
        else:
            print('ERROR: unknown channel {}'.format(settings.channel))

        signal = np.ascontiguousarray(signal.T.astype(np.int16))
        self.__sound = pygame.sndarray.make_sound(signal)
    
    def play(self):
        self.__sound.play()


class OutputHandler:
    def __init__(self, intervals):
        self.__display = FullDisplay(playedFrequency=settings.neuronalFrequency,
                                     frequencies=settings.presynapticFrequencies,
                                     intervals=intervals,
                                     types=list(settings.presynapticNeurons.values()),
                                     width=settings.displaySize[0],
                                     height=settings.displaySize[1])
        
        self.__player = SoundPlayer()
            
    def update(self):
        self.__display.update()
        if valueHandler['hasSpiked']:
            self.__player.play()
    
    def forcePlay(self):
        self.__player.play()
    
    def toggleFullscreen(self):
        self.__display.toggleFullscreen()


class ThreadRecorder(Thread):
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    SWIDTH = pyaudio.get_sample_size(FORMAT)

    def __init__(self, input_name='Built-in Microphone', channel_id=1, refresh_interval=0.05):
        super(ThreadRecorder, self).__init__()
        self.__refresh_interval = refresh_interval  # s
        self.__nread = int(self.__refresh_interval * SoundPlayer.RATE)
        self.__p = pyaudio.PyAudio()
        self.__create_stream(input_name, channel_id)
        self.__is_recording = True
        self.__send_to_engine = None
        
    def __create_stream(self, device_name, channel_id):
        index = self.__get_index_by_name(device_name)
        if index >= 0:
            self.__stream = self.__p.open(format=self.FORMAT,
                                          channels=channel_id,
                                          rate=SoundPlayer.RATE,
                                          input=True,
                                          input_device_index=index,
                                          frames_per_buffer=self.CHUNK)
            print(('SETUP input:', device_name, 'connected'))
        else:
            ''' SEND TO STOUT? '''
            print(('no such input device', device_name))
        
    def __get_index_by_name(self, device_name):
        n = self.__p.get_device_count()
        index = -1
        for k in range(n):
            inf = self.__p.get_device_info_by_index(k)
            if inf['name'] == device_name and inf['maxInputChannels'] > 0:
                index = inf['index']
            return index
        return index
    
    def setEngineCb(self, engine_cb):
        self.__send_to_engine = engine_cb
    
    def run(self):  # has to be named 'run', because Thread.start() calls 'run'!!!
        while self.__is_recording:
            data = None
            try:
                raw_data = self.__stream.read(self.__nread)
                data = np.array(wave.struct.unpack("%dh" % (len(raw_data)/self.SWIDTH), raw_data))
            except IOError:
                pass
            if data is not None:
                self.__send_to_engine(data)
                
    def stop(self):
        self.__is_recording = False


class Recorder:
    SWIDTH = pyaudio.get_sample_size(ThreadRecorder.FORMAT)

    def __init__(self, inputName='Microphone', channelId=1, refreshInterval = 0.05):
        self.__refreshInterval = refreshInterval  # s
        self.__nread = int(self.__refreshInterval * SoundPlayer.RATE)
        self.__createStream(inputName, channelId)
        self.__sendToEngine = None
        
    def setEngineCb(self,engineCb):
        self.__sendToEngine = engineCb 
        
    def __createStream(self, inputName, channelId):
        self.__stream = p.open(format=ThreadRecorder.FORMAT,
                               channels=channelId,
                               rate=SoundPlayer.RATE,
                               input=True,
                               input_device_index=self.__getIndexByName(inputName),
                               frames_per_buffer=ThreadRecorder.CHUNK)
        
    def __getIndexByName(self, inputName):
        return 0
    
    def record(self):
        nbits = self.__stream.get_read_available()
        try:
            data = self.__stream.read(self.__nread, exception_on_overflow=False)  # TODO catch proper exception
            realdata = np.array(wave.struct.unpack("%dh" % (len(data)/self.SWIDTH), data))
            self.__sendToEngine(realdata)
        except OSError as ex:
            print(('skipping audio', nbits))
            data = self.__stream.read(self.__nread, exception_on_overflow=False)
            realdata = np.array(wave.struct.unpack("%dh" % (len(data)/self.SWIDTH), data))
            self.__sendToEngine(realdata)


class FrequencyDetector:
    def __init__(self, frequencies=None, threshold=150, tolerance=.1):
        self.__frequencies = frequencies 
        self.__frequencyIntervals = []
        self.__createFrequencyIntervals(frequencies, tolerance)
        self.__threshold = threshold
        self.__timeDetected = np.zeros_like(frequencies)
        
    @property
    def intervals(self):
        return self.__frequencyIntervals

    def get(self ,xData, fftData):
        return self.__detect(xData, fftData)
    
    def __createFrequencyIntervals(self, freqs, tol):
        lower = (1 - tol * (1-1/settings.NOTERATIO))
        upper = (1 + tol * (settings.NOTERATIO-1))
        [self.__frequencyIntervals.append([freq * lower, freq * upper]) for freq in freqs]
    
    def __detect(self, xData, fftData):
        actList = len(self.__frequencies)*[True]
        for id, (mn, mx) in enumerate(self.__frequencyIntervals):
            if time.time() - self.__timeDetected[id] > settings.toneDuration/1000.:
                idx, = np.nonzero((xData >= mn) & (xData <= mx))
                volume = np.mean(fftData[idx])
                if volume > self.__threshold:
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
    def __init__(self, minFreq=1e2, maxFreq=1e4):
        self.__minFreq=minFreq
        self.__maxFreq=maxFreq

    def get(self, x_data, fft_data):
        return self.__detect(x_data, fft_data)
    
    def __detect(self, x_data, fft_data):
        return x_data[np.argmax(fft_data)]
    

class InputEngine:
    __outputCb = None
    __neuron = None

    def __init__(self, recorder=Recorder(refreshInterval=0.1)):
        self.attach(1)
        self.__plot = plot
        self.__detector = FrequencyDetector(frequencies=settings.presynapticFrequencies,
                                            tolerance=settings.frequencyTolerance,
                                            threshold=settings.frequencyThreshold)
        self.__recorder = recorder
        self.__recorder.setEngineCb(self.__onAudioReceive)
        
    @property
    def intervals(self):
        return self.__detector.intervals
    
    def setOutputCb(self, pyfunc):
        self.__outputCb = pyfunc
        
    def update(self):
        self.__recorder.record()
        
    def attach(self, neuronId):
        """connect a neuron to this Microphone, adding the freqs and the update Callback"""
        self.__neuron = DestexheNeuron()
        pars = settings.defaultPars(settings.neuronalType)
        pars.update(settings.neuronParameters)
        valueHandler.update(**pars)
        self.__neuron.setParams(presynapticNeurons=settings.presynapticNeurons, **pars)
        
    def __onAudioReceive(self, data):
        if len(data) > 1:
            fftData = np.fft.fft(data)/data.size
            fftData = np.abs(fftData[list(range(int(data.size/2)))])
            frqs = np.arange(data.size)/(data.size / float(SoundPlayer.RATE))
            xData = frqs[list(range(int(data.size/2)))]
            valueHandler.update(xData=xData, fftData=fftData)
            detectedFreqs = self.__detector.get(xData, fftData)
            valueHandler.update(detectedFreqs=detectedFreqs)
            vals = self.__neuron.update()

            ''' SEND NEXT LINE TO A SUBPROCESS / MULTIPROCESS ! '''
            self.__outputCb()#(xData,fftData,self.__neuron._v,vals)


class MainApp:
    def __init__(self):
        pygame.init()
        self.__inputEngine = InputEngine()
        self.__outputHandler = OutputHandler(self.__inputEngine.intervals)
        self.__inputEngine.setOutputCb(self.__outputHandler.update)
        self.__fullscreen = False
        
    def input(self, events):
        for event in events: 
            if event.type == pygame.locals.QUIT:
                sys.exit(0)
            elif event.type == pygame.locals.MOUSEBUTTONDOWN:
                self.__outputHandler.forcePlay()
            elif event.type == pygame.locals.KEYDOWN:
                if event.dict['key'] == pygame.locals.K_p:
                    self.__outputHandler.forcePlay()
                elif event.dict['key'] == pygame.locals.K_ESCAPE:
                    sys.exit(0)
                elif event.dict['key'] == pygame.locals.K_f:
                    self.__fullscreen = not self.__fullscreen
                    if self.__fullscreen:
                        pygame.display.set_mode(settings.displaySize, pygame.locals.FULLSCREEN)
                    else:
                        pygame.display.set_mode(settings.displaySize)
    
    def run(self):
        upd_int = .01
        now = time.time()
        while True:
            self.input(pygame.event.get())
            if time.time()-now >= upd_int:
                self.__inputEngine.update()
                now = time.time()


if __name__=='__main__':
    
    plot = True
    if len(sys.argv)>1:
        print((sys.argv))
        if sys.argv[-1] == 'p':
            plot = True
    app = MainApp()
    app.run()
    
#    input = InputEngine(plot=plot)
#    raw_input('los?')