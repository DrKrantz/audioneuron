# the caller of the neuron-updater

import math
import time


class Synapse(object):
    def __init__(self, s=6e-10):
        super(Synapse, self).__init__()
        self.strength = s
        
    def update(self, presynaptic_spike=True):
        if presynaptic_spike:
            return self.strength
        else:
            return 0


class DestexheNeuron(object):
    def __init__(self, **kwargs):
        super(DestexheNeuron, self).__init__()
        self.setParams(**kwargs)

        #  initialize remaining values
        self._remainingDeadtime = 0
        self._w = 0
        self._ge = 0
        self._gi = 0
        self._hasSpiked = False
        self.__lastUpdate = time.time()
        self.recording = {'ge': 'ge', 'gi': 'gi', 'v': 'v', 'spikes': 'spikes', 'w': 'w'}
        self.recordingActive = False
        self.__hasRecorded = False
        self._setupSynapses()
    
    def setParams(self,threshold = -50e-3, Cm=1e-6, gL=0.05e-3, EL=-60e-3,
                Delta=2.5e-3,
                S=20000e-8, dead=2.5e-3, a=0.08e-6, b=0, tau_w=600e-3,
                Ee=0e-3, Ei=-80e-3,
                s_e=6e-10, s_i=67e-9, tau_e=5e-3, tau_i=10e-3,
                external=0, presynapticNeurons={}, maxMspTime=20./1, port=0):
        
        self._threshold = threshold  # firing threshold of individual neurons, V
        self._Cm = Cm  # membrane capacitance, F/cm2
        self._gL = gL  # leak conductance, S/cm2
        self._EL = EL  # resting potential = reset after spike, V
        self._Delta = Delta  # steepness of exponential, V
        self._S = S  # membrane area, cm2
        self._dead = dead  # dead time
        
        self._a = a  # adaptation dynamics of the synapses, S
        self._b = b  # increment of adaptation after a spike
        self._tau_w = tau_w  # time-constant of adaptation variable, sec
        self._Ee = Ee  # reversal potential of excitatory synapses, V
        self._Ei = Ei  # reversal potential of inhibitory synapses, V
        self._external = external
        self._v = EL

        self._s_e = s_e  # increment of excitatory synaptic conductance per spike, S,
        self._s_i = s_i  # increment of inhibitory synaptic conductance S per spike
        self._tau_e = tau_e  #sec,
        self._tau_i = tau_i  #sec
        self._presynapticNeurons = presynapticNeurons  #{freq: neuronType (E/I)}
        self._maxMspTime = maxMspTime  # 1 sec in python corresponds to maxMspTime secs in Max/MSP
        self._port = port
        self._runtime = 0
        self._setupSynapses()
    
    def switchRecordingState(self):
        if not self.recordingActive:
            self.recordingActive = True
        else:
            self.recordingActive = False
            self.__writeRecording()
    
    def _setupSynapses(self):
        self._synapses = {}
        for presNeuron, type in self._presynapticNeurons.items():
            if type == 'E':
                self._synapses[presNeuron] = Synapse(s=self._s_e)
            else:
                self._synapses[presNeuron] = Synapse(s=self._s_i)
        
    def __recordVariables(self):
        if self.recordingActive:
            recordingAccuracy = 1e5
#            if (self._runtime<2) and not(self.__hasRecorded):
            self.recording['v'] += ', '+str(self._v)  #str(round(recordingAccuracy*self._v)/recordingAccuracy)
            self.recording['ge'] += ', '+str(self._ge)
            self.recording['gi'] += ', '+str(self._gi)
            self.recording['spikes'] += ', '+str(int(self._hasSpiked))
            self.recording['w'] += ', '+str(self._w)
    
    def __writeRecording(self):
        tordir = '/Volumes/data1/__Gastkuenstler__/TimOttoRoth/TimOttoRoth_August2011/Neuron/data/'
        output = open(tordir+'data_port'+str(int(self._port))+'.txt', 'w')
        output.write(self.recording['v']+'\n')
        output.write(self.recording['ge']+'\n')
        output.write(self.recording['gi']+'\n')
        output.write(self.recording['w']+'\n')
        output.write(self.recording['spikes'])
        output.close()
        print(('parameters for port', int(self._port), 'saved'))
        output.close()
        self.__hasRecorded = True
         
    def update(self, detected_frequencies):
        """
        TODO
        # COMPUTE dt!!!
        realdt = (time.time()-self.__lastUpdate)
        dt = realdt/self._maxMspTime
        self.__lastUpdate = time.time()
        """
        dt = 1e-3
        self._runtime += dt

        self._update_conductances(detected_frequencies, dt)
        self._update_membrane(dt)
        return self._hasSpiked

    def get_value(self, value):
        return getattr(self, "_" + value)
    
    def _update_membrane(self, dt):
        if self._v >= self._threshold:
            self._v = self._EL  # set spiked neurons to reset potential

        self._hasSpiked = False
        if self._remainingDeadtime > 0:
            self._remainingDeadtime -= dt
        else:
            # UPDATE MEMBRANE POTENTIAL, ADAPTATION AND CHECK FOR SPIKE
            v = self._v + dt*(-self._gL*(self._v-self._EL) +
                              self._gL*self._Delta*math.exp((self._v-self._threshold)/self._Delta)
                              - self._w/self._S
                              - self._ge*(self._v-self._Ee)/self._S
                              - self._gi*(self._v-self._Ei)/self._S
                              )/self._Cm
            self._v = v + dt*self._external/self._Cm
            self._w += dt*(self._a*(self._v-self._EL)-self._w)/self._tau_w
            if self._v >= self._threshold:
                self._remainingDeadtime = self._dead
                self._w += 0  # self._b #increment adaptation variable of spiked neurons
                self._hasSpiked = True
                
        self.__recordVariables()
            
    def _update_conductances(self, detected_frequencies, dt):
        input_e, input_i = 0, 0
        for neuron_id, synapse in self._synapses.items():
            if self._presynapticNeurons[neuron_id] == 'E':
                input_e += synapse.update(presynaptic_spike=detected_frequencies[neuron_id])
            else:
                input_i += synapse.update(presynaptic_spike=detected_frequencies[neuron_id])
        self._ge += -dt*self._ge/self._tau_e+input_e
        self._gi += -dt*self._gi/self._tau_i+input_i
