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

valueHandler = valuehandler.ValueHandler()
p = pyaudio.PyAudio()


def get_index_by_name(input_name):
    n = p.get_device_count()
    index = -1
    for k in range(n):
        inf = p.get_device_info_by_index(k)
        if inf['name'] == input_name and inf['maxInputChannels'] > 0:
            index = inf['index']
        return index
    return index


class SoundPlayer:
    CHANNELS = 2

    def __init__(self):
        self.__sound = pygame.mixer.Sound(np.array([]))
        self.__create_sound()
        self.__channel = self.__sound.play()
    
    def __create_sound(self):
        num_samples = int(settings.sampling_rate*settings.toneDuration/1000.)
        bits = 16
        twopi = 2*np.pi

        pygame.mixer.pre_init(44100, -bits, self.CHANNELS)
        pygame.init()

        max_sample = 2 ** (bits - 1) - 1
        sine = max_sample * np.sin(np.arange(num_samples)*twopi*settings.neuronalFrequency/settings.sampling_rate)

        #  add an- u- abschwellen
        if settings.dampDuration < settings.toneDuration / 2.:
            num_damp_samp = int(settings.sampling_rate * settings.dampDuration)
        else:
            print('WARNING! dampDuration>toneDuration/2!!! Using toneDuration/2')
            num_damp_samp = int(settings.sampling_rate * settings.toneDuration / 2.)  # the 2 is the stereo
        damp_vec = np.linspace(0, 1, num_damp_samp)
        sine[0:num_damp_samp] *= damp_vec
        sine[len(sine) - num_damp_samp::] *= damp_vec[::-1]

        #  add ending silence
        sine = np.concatenate([sine, np.zeros(int(settings.sampling_rate * settings.endSilence))])

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
        self.__channel = self.__sound.play()

    def get_busy(self):
        return self.__channel.get_busy()


class ThreadRecorder(Thread):
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    SWIDTH = pyaudio.get_sample_size(FORMAT)

    def __init__(self, input_name='Built-in Microphone', channel_id=1, refresh_interval=0.05):
        super(ThreadRecorder, self).__init__()
        self.__refresh_interval = refresh_interval  # s
        self.__nread = int(self.__refresh_interval * SoundPlayer.RATE)
        self.__create_stream(input_name, channel_id)
        self.__is_recording = True
        self.__send_to_engine = None
        
    def __create_stream(self, device_name, channel_id):
        index = get_index_by_name(device_name)
        if index >= 0:
            self.__stream = p.open(format=self.FORMAT,
                                   channels=channel_id,
                                   rate=SoundPlayer.RATE,
                                   input=True,
                                   input_device_index=index,
                                   frames_per_buffer=self.CHUNK)
            print(('SETUP input:', device_name, 'connected'))
        else:
            ''' SEND TO STOUT? '''
            print(('no such input device', device_name))
    
    def set_engine_cb(self, engine_cb):
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

    def __init__(self, input_name='Microphone', channel_id=1):
        self.__create_stream(input_name, channel_id)
        
    def __create_stream(self, input_name, channel_id):
        self.__stream = p.open(format=ThreadRecorder.FORMAT,
                               channels=channel_id,
                               rate=settings.sampling_rate,
                               input=True,
                               input_device_index=get_index_by_name(input_name),
                               frames_per_buffer=ThreadRecorder.CHUNK)
        
    def record(self):
        nbits = self.__stream.get_read_available()
        try:
            raw_data = self.__stream.read(settings.recording_chunk_size, exception_on_overflow=False)  # TODO catch proper exception
            data = np.array(wave.struct.unpack("%dh" % (len(raw_data) / self.SWIDTH), raw_data))
        except OSError:
            print(('skipping audio', nbits))
            raw_data = self.__stream.read(settings.recording_chunk_size, exception_on_overflow=False)
            data = np.array(wave.struct.unpack("%dh" % (len(raw_data) / self.SWIDTH), raw_data))
        return data


class FrequencyDetector:
    def __init__(self, frequencies=None, threshold=150, tolerance=.1):
        self.__frequencies = frequencies 
        self.__frequencyIntervals = []
        self.__create_frequency_intervals(frequencies, tolerance)
        self.__threshold = threshold
        self.__time_detected = np.zeros_like(frequencies)
        
    def get_intervals(self):
        return self.__frequencyIntervals
    
    def __create_frequency_intervals(self, freqs, tol):
        lower = (1 - tol * (1-1/settings.NOTERATIO))
        upper = (1 + tol * (settings.NOTERATIO-1))
        [self.__frequencyIntervals.append([freq * lower, freq * upper]) for freq in freqs]
    
    def detect(self, x_data, fft_data):
        detected_freqs = len(self.__frequencies)*[True]
        for freq_id, (mn, mx) in enumerate(self.__frequencyIntervals):
            if time.time() - self.__time_detected[freq_id] > settings.toneDuration/1000.:
                idx, = np.nonzero((x_data >= mn) & (x_data <= mx))
                volume = np.mean(fft_data[idx])
                if volume > self.__threshold:
                    self.__time_detected[freq_id] = time.time()
                else:
                    detected_freqs[freq_id] = False
            else:
                detected_freqs[freq_id] = False
        if np.any(detected_freqs):
            idx, = np.nonzero(detected_freqs)
            print(('detected:', np.array(self.__frequencies)[idx]))
        return detected_freqs


class AudioSynapse:
    def __init__(self, frequency: float, tolerance=0.1, threshold=150):
        self.frequency = frequency
        self.__threshold = threshold
        self.lower_freq = 0
        self.upper_freq = 0
        self.__sum_idx = self.__create_indices(tolerance)
        self.__time_detected = 0

    def __create_indices(self, tolerance: float) -> list:
        """
        collect the indices that are in the tolerance range around the frequency
        :param tolerance:
        :return:
        """
        self.lower_freq = (1 - tolerance * (1 - 1 / settings.NOTERATIO)) * self.frequency
        self.upper_freq = (1 + tolerance * (settings.NOTERATIO - 1)) * self.frequency

        frqs = np.arange(settings.recording_chunk_size) / (settings.recording_chunk_size/float(settings.sampling_rate))
        x_data = frqs[list(range(int(settings.recording_chunk_size / 2)))]

        sum_idx = []
        for idx, freq in enumerate(x_data):
            if self.lower_freq <= freq <= self.upper_freq:
                sum_idx.append(idx)
            elif freq > self.upper_freq:
                break
        return sum_idx

    def detect(self, current_signal: list) -> bool:
        """
        returns True if the AVERAGE signal in the frequency band of the synapse is above the threshold
        :param current_signal:
        :return:
        """
        if time.time() - self.__time_detected < settings.toneDuration/1000.:
            return False

        volume = 0
        for idx in self.__sum_idx:
            volume += current_signal[idx]

        if (volume / len(self.__sum_idx)) < self.__threshold:
            return False

        self.__time_detected = time.time()
        print('detected:', self.frequency)
        return True


class SynapticAudioTree:
    def __init__(self, frequencies: list, tolerance=0.1, threshold=150):
        self.__tolerance = tolerance
        self.__synapses = []
        [self.__synapses.append(AudioSynapse(frequency, tolerance, threshold)) for frequency in frequencies]

    def detect(self, signal: list) -> list:
        synapses_active = []
        [synapses_active.append(synapse.detect(signal)) for synapse in self.__synapses]
        return synapses_active

    def get_intervals(self) -> list:
        """
        Return the frequency bands that are detected from the synapses
        :return:
        """
        intervals = []
        [intervals.append([synapse.lower_freq, synapse.upper_freq]) for synapse in self.__synapses]
        return intervals


class MainApp:
    def __init__(self):
        pygame.init()
        self.__fullscreen = False
        self.__recorder = Recorder()
        self.__detector = SynapticAudioTree(settings.presynapticFrequencies,
                                            settings.frequencyTolerance,
                                            settings.frequencyThreshold)

        self.__neuron = DestexheNeuron()
        pars = settings.defaultPars(settings.neuronalType)
        self.__neuron.setParams(presynapticNeurons=settings.presynapticNeurons, **pars)

        self.__display = FullDisplay(playedFrequency=settings.neuronalFrequency,
                                     frequencies=settings.presynapticFrequencies,
                                     intervals=self.__detector.get_intervals(),
                                     types=list(settings.presynapticNeurons.values()),
                                     threshold=pars['threshold'],
                                     resting_potential=pars['EL'],
                                     width=settings.displaySize[0],
                                     height=settings.displaySize[1])

        self.__player = SoundPlayer()
        
    def input(self, events):
        for event in events: 
            if event.type == pygame.locals.QUIT:
                sys.exit(0)
            elif event.type == pygame.locals.MOUSEBUTTONDOWN:
                self.__player.play()
            elif event.type == pygame.locals.KEYDOWN:
                if event.dict['key'] == pygame.locals.K_p:
                    self.__player.play()
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
            time.sleep(0.001)
            if time.time()-now >= upd_int:
                now = time.time()
                if not self.__player.get_busy():
                    data = self.__recorder.record()
                    fft_data = self.__fft(data)
                    detected_freqs = self.__detector.detect(fft_data)
                    has_fired = self.__neuron.update(detected_freqs)

                    draw_values = dict(y=fft_data, v=self.__neuron.get_value('v'),
                                       detected_freqs=detected_freqs)
                    self.__display.update(has_fired, **draw_values)

                    if has_fired:
                        self.__player.play()

    @staticmethod
    def __fft(data):
        fft_data = np.fft.fft(data) / data.size
        fft_data = np.abs(fft_data[list(range(int(data.size / 2)))])
        return fft_data


if __name__ == '__main__':
    plot = True
    if len(sys.argv) > 1:
        print(sys.argv)
        if sys.argv[-1] == 'p':
            plot = True
    app = MainApp()
    app.run()
