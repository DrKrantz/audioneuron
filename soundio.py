import pygame
import pyaudio
import wave
import numpy as np
import settings

p = pyaudio.PyAudio()


def fft(data):
    fft_data = np.fft.fft(data) / data.size
    fft_data = np.abs(fft_data[list(range(int(data.size / 2)))])
    return fft_data


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
        num_samples = int(settings.sampling_rate * settings.toneDuration / 1000.)
        bits = 16
        twopi = 2 * np.pi

        pygame.mixer.pre_init(44100, -bits, self.CHANNELS)
        pygame.init()

        max_sample = 2 ** (bits - 1) - 1
        sine = max_sample * np.sin(np.arange(num_samples) * twopi * settings.neuronalFrequency / settings.sampling_rate)

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


class Recorder:
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    SWIDTH = pyaudio.get_sample_size(pyaudio.paInt16)

    def __init__(self, input_name='Microphone', channel_id=1):
        self.__create_stream(input_name, channel_id)

    def __create_stream(self, input_name, channel_id):
        self.__stream = p.open(format=Recorder.FORMAT,
                               channels=channel_id,
                               rate=settings.sampling_rate,
                               input=True,
                               input_device_index=get_index_by_name(input_name),
                               frames_per_buffer=Recorder.CHUNK)

    def record(self):
        nbits = self.__stream.get_read_available()
        try:
            raw_data = self.__stream.read(settings.recording_chunk_size,
                                          exception_on_overflow=False)  # TODO catch proper exception
            data = np.array(wave.struct.unpack("%dh" % (len(raw_data) / self.SWIDTH), raw_data))
        except OSError:
            print(('skipping audio', nbits))
            raw_data = self.__stream.read(settings.recording_chunk_size, exception_on_overflow=False)
            data = np.array(wave.struct.unpack("%dh" % (len(raw_data) / self.SWIDTH), raw_data))
        return data
