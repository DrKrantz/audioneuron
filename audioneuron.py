import sys
import time

from neuroncontrol import DestexheNeuron
import settings
import display
from soundio import SoundPlayer, Recorder, fft


x_data = []
[x_data.append(idx * settings.sampling_rate/settings.recording_chunk_size)
 for idx in range(int(settings.recording_chunk_size/2))]


def compute_frequency_indices(lower: float, upper: float) -> list:
    """
    collect the indices that are in the tolerance range around the frequency
    :param lower:
    :param upper:
    :return:
    """
    sum_idx = []
    for idx, freq in enumerate(x_data):
        if lower <= freq <= upper:
            sum_idx.append(idx)
        elif freq > upper:
            break
    return sum_idx


class AudioSynapse:
    def __init__(self, frequency: float, tolerance=0.1, threshold=150):
        self.frequency = frequency
        self.lower_freq = 0
        self.upper_freq = 0
        self.__threshold = threshold
        self.__sum_idx = self.__create_indices(tolerance)
        self.__time_detected = 0

    def __create_indices(self, tolerance: float) -> list:
        self.lower_freq = (1 - tolerance * (1 - 1 / settings.NOTERATIO)) * self.frequency
        self.upper_freq = (1 + tolerance * (settings.NOTERATIO - 1)) * self.frequency
        return compute_frequency_indices(self.lower_freq, self.upper_freq)

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
        self.__recorder = Recorder()
        self.__detector = SynapticAudioTree(settings.presynapticFrequencies,
                                            settings.frequencyTolerance,
                                            settings.frequencyThreshold)

        self.__neuron = DestexheNeuron()
        pars = settings.defaultPars(settings.neuronalType)
        self.__neuron.setParams(presynapticNeurons=settings.presynapticNeurons, **pars)

        self.__display = display.FullDisplay(playedFrequency=settings.neuronalFrequency,
                                             frequencies=settings.presynapticFrequencies,
                                             intervals=self.__detector.get_intervals(),
                                             types=list(settings.presynapticNeurons.values()),
                                             threshold=pars['threshold'],
                                             resting_potential=pars['EL'],
                                             width=settings.displaySize[0],
                                             height=settings.displaySize[1])

        self.__player = SoundPlayer()
    
    def run(self):
        upd_int = .01
        now = time.time()
        while True:
            value = self.__display.get_keyboard_input()
            if value == display.COMMAND_PLAY:
                self.__player.play()

            time.sleep(0.001)
            if time.time()-now >= upd_int:
                now = time.time()
                if not self.__player.get_busy():
                    data = self.__recorder.record()
                    fft_data = fft(data)
                    detected_freqs = self.__detector.detect(fft_data)
                    has_fired = self.__neuron.update(detected_freqs)

                    draw_values = dict(y=fft_data, v=self.__neuron.get_value('v'),
                                       detected_freqs=detected_freqs)
                    self.__display.update(has_fired, **draw_values)

                    if has_fired:
                        self.__player.play()


if __name__ == '__main__':
    plot = True
    if len(sys.argv) > 1:
        print(sys.argv)
        if sys.argv[-1] == 'p':
            plot = True
    app = MainApp()
    app.run()
