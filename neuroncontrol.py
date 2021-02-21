# the caller of the neuron-updater

import math
import time
import numpy as np
import valuehandler
valueHandler = valuehandler.ValueHandler()


class Synapse(object):
    def __init__(self, s=6e-10):
        super(Synapse, self).__init__()
        self.strength = s
        
    def update(self, presynapticSpike=True):
        if presynapticSpike:
            return self.strength
        else:
            return 0


class PlasticSynapse(Synapse):
    def __init__(self, tauSTDP=0.02, Aplus=1e-9, Aminus=0, **kwargs):
        super(PlasticSynapse,self).__init__(**kwargs)
        self.tauSTDP = tauSTDP
        self.Aplus = Aplus
        self.Aminus = Aminus
        self.lastActive = 0
    
    def setLastSpikeGetter(self, lastSpikeGetter):
        self.__getlastSpike = lastSpikeGetter
    
    def update(self, time, presynapticSpike=True, postsynapticSpike=True):
        if presynapticSpike and not postsynapticSpike:
            self.lastActive=time
            self.strength -= self.Aminus*math.exp(
                            (self.lastActive-self.__getlastSpike())/self.tauSTDP) 
            returnValue = self.strength
        elif not presynapticSpike and postsynapticSpike:
            self.strength +=  self.Aplus*math.exp((time-self.lastActive)/self.tauSTDP)
            returnValue = 0
#            print self.strength
        else:
            self.strength += 0
            returnValue = 0
        return returnValue


class Neuron(object):
    def __init__(self):
        super(Neuron,self).__init__()
        self._port = 0
        self._v = 0

        self._gL = 0  # leak conductances, S/cm2
        self._threshold = 0  # firing threshold of individual neurons, V
        self._Cm = 0  # membrane capacitance, F/cm2
        self._gL = 0   # leak conductances, S/cm2
        self._EL = 0  # resting potential = reset after spike, V
        self._Delta = 0  # steepness of exponential, V
        self._S = 0  # membrane area, cm2
        self._dead = 0  # deadtime
        self._external = 0
        self._hasSpiked = False
    
    def update(self):
        pass
    

class DestexheNeuron(object):
    def __init__(self,**kwargs):
        super(DestexheNeuron,self).__init__()
        self.setParams(**kwargs)

        #  initialize reamining values
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
        self._gL = gL  # leak conductances, S/cm2
        self._EL = EL  # resting potential = reset after spike, V
        self._Delta = Delta  # steepness of exponential, V
        self._S = S  # membrane area, cm2
        self._dead = dead  # deadtime
        
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
        for presNeuron, type in list(self._presynapticNeurons.items()):
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
        TORDIR = '/Volumes/data1/__Gastkuenstler__/TimOttoRoth/TimOttoRoth_August2011/Neuron/data/'
        output = open(TORDIR+'data_port'+str(int(self._port))+'.txt', 'wb')
        output.write(self.recording['v']+'\n')
        output.write(self.recording['ge']+'\n')
        output.write(self.recording['gi']+'\n')
        output.write(self.recording['w']+'\n')
        output.write(self.recording['spikes'])
        output.close()
        print(('parameters for port', int(self._port), 'saved'))
        output.close()
        self.__hasRecorded = True
         
    def update(self):
        activeNeurons, = np.nonzero(valueHandler['detectedFreqs'])
        self._updateMembrane(activeNeurons)
        factor = 1
        valueHandler.update(hasSpiked = self._hasSpiked, v = self._v,
                            ge=factor*self._ge, gi=factor*self._gi,
                            w=factor*self._w)
        return self._hasSpiked,self._v, factor*self._ge, factor*self._gi, factor*self._w
    
    def _updateMembrane(self, activeNeurons):
        '''
        # COMPUTE dt!!!
        realdt = (time.time()-self.__lastUpdate)
        dt = realdt/self._maxMspTime
        self.__lastUpdate = time.time()
        '''
        if self._v>=self._threshold:
            self._v = self._EL  # set spiked neurons to reset potential
        dt = 1e-3
        self._runtime += dt
        self._updateConductances(activeNeurons, dt)
        self._hasSpiked = False
        if self._remainingDeadtime>0:
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
            
    def _updateConductances(self, activeNeurons, dt):
        input_e, input_i = 0,0
        for neuron, synapse in list(self._synapses.items()):
            if self._presynapticNeurons[neuron]=='E':
                input_e += synapse.update(presynapticSpike=neuron in activeNeurons)
            else:
                input_i += synapse.update(presynapticSpike=neuron in activeNeurons)
        self._ge += -dt*self._ge/self._tau_e+input_e
        self._gi += -dt*self._gi/self._tau_i+input_i
        

class PlasticDestexheNeuron(DestexheNeuron):
    def __init__(self, **kwargs):
        super(PlasticDestexheNeuron, self).__init__(**kwargs)
        self.__lastSpikeTime = 0
        
    def __lastSpikeGetter(self):
        return self.__lastSpikeTime
    
    def _setupSynapses(self):
        self._synapses = {}
        for presNeuron, n_type in list(self._presynapticNeurons.items()):
            if n_type == 'E':
                self._synapses[presNeuron] = PlasticSynapse(s=self._s_e)
            else:
                self._synapses[presNeuron] = PlasticSynapse(s=self._s_i)
            self._synapses[presNeuron].setLastSpikeGetter(self.__lastSpikeGetter)

    def _updateConductances(self, activeNeurons, dt):
        input_e, input_i = 0, 0
        for presNeuron, synapse in list(self._synapses.items()):
            if self._presynapticNeurons[presNeuron] == 'E':
                input_e += synapse.update(self._runtime,
                                          presynapticSpike=presNeuron in activeNeurons,
                                          postsynapticSpike=self._hasSpiked)
            else:
                input_i += synapse.update(self._runtime,
                                          presynapticSpike=presNeuron in activeNeurons,
                                          postsynapticSpike=self._hasSpiked)
        self._ge += -dt*self._ge/self._tau_e + input_e
        self._gi += -dt*self._gi/self._tau_i + input_i
    
    def update(self, activeNeurons):
        self._updateMembrane(activeNeurons)
        if self._hasSpiked:
            self.__lastSpikeTime = self._runtime
        return self._hasSpiked, self._v

def osclist2Dict(inputString):
    args = ['port','threshold','Cm', 'gL', 'EL','Delta','S','dead', 'a', 'b', 'tau_w', 
                            'Ee', 'Ei','s_e', 's_i', 'tau_e', 'tau_i','external',
                            'presynapticNeurons','presynapticTypes','maxMspTime',
                            'toneDuration',
                            'type']
    inputList = inputString
    parDict = dict.fromkeys(args)
    key = 'trash'
    
    val = []
    for entry in inputList:
        if entry in args:
            parDict[key] = val[0] if len(val) == 1 else val
            key = entry
            val = []
        else:
            if entry in ['E', 'I']:
                val.append(entry)
            else:
                val.append(float(entry))
    parDict[key] = val[0] if len(val) == 1 else val
    parDict.pop('trash')
    return parDict


def splitParDict(parDict):
    # construct the liste that's sent back to Max
    maxArgs = ['toneDuration', 'presynapticNeurons']
    
    maxList = ['toneDuration', int(parDict['toneDuration']*1000), 'presynapticNeurons']
    
    if isinstance(parDict['presynapticNeurons'], list):
        for presNeuron in parDict['presynapticNeurons']:
            maxList.append(int(presNeuron))
    elif isinstance(parDict['presynapticNeurons'], float):
        maxList.append(int(parDict['presynapticNeurons']))
    # construct the dictionary to initialize the neuron
    neuronArgs = ['port', 'threshold', 'Cm', 'gL', 'EL', 'Delta', 'S', 'dead', 'a', 'b', 'tau_w',
                            'Ee', 'Ei', 's_e', 's_i', 'tau_e', 'tau_i', 'external',
                            'presynapticNeurons', 'presynapticTypes', 'maxMspTime']
    neuronDict = parDict
    for key in ['maxMspTime', 'toneDuration', 'type']:
            neuronDict.pop(key)
    # reformat presynapticFrequencies as {freq:type,...}-dict
    presynapticNeurons = {}
    for freq, type in zip(neuronDict['presynapticNeurons'], neuronDict.pop('presynapticTypes')):
            presynapticNeurons.setdefault(freq, type)
    neuronDict['presynapticNeurons'] = presynapticNeurons 

    return neuronDict, maxList


def fromMax(inlet, *inputList):
    if isinstance(inputList[0], str):
        parDict = osclist2Dict(inputList)
        neuronDict ,maxList = splitParDict(parDict)
        neuron.setParams(**neuronDict)
        maxObject.outlet(0, maxList)
    else:
#        print 'es kommt an:', inputList
        result = neuron.update(inputList)
#        print result
        maxObject.outlet(0, result)
        
def _int(inlet, *activeNeurons):
    if activeNeurons[0] > 200:
        result = neuron.update([])#activeNeurons
        maxObject.outlet(0, result)
#        print 'hier hamwa den Salat!', activeNeurons
    else:
#        print 'JETZT PASSIERT\'S!!!'
        result = neuron.update([])#activeNeurons
        maxObject.outlet(0, result)
    
def bang(input):
    pass

def record(input):
    neuron.switchRecordingState()

'''
class Test(DestexheNeuron):
    def __init__(self,nSteps=1000,**kwargs):
        super(Test,self).__init__(s_e = 5e-9,
                    presynapticNeurons={1:'E',2:'E',3:'I'})
        from parameters import biolPars
        pars = biolPars()
        neuronArgs = ['threshold','Cm', 'gL', 'EL','Delta','S','dead', 'tau_w', 
                            'Ee', 'Ei','s_e', 's_i', 'tau_e', 'tau_i','external']
        unusedKeys = set(pars.keys()).difference(neuronArgs)
        neuronDict = {}
        [pars.pop(key) for key in unusedKeys]
        
#        print biolPars.keys()
        self.setParams(**pars)
        self.__vMem = []
        self.nSteps = nSteps
#        self._tau_e = 5e-3 #sec,
#        self._tau_i = 10e-3 #sec
#        self._v=-55e-3
#        self._external = 7e-7

    def start(self,ge=0,gi=0):
        vMem = []
        geMem = []
        giMem = []
        for k in range(self.nSteps):
            vMem.append(self._v)
            geMem.append(self._ge)
            giMem.append(self._gi)
#            time.sleep(0.0001)
            freq = []
            if k in [200,201,400]:
                freq = [123]
#            if k in range(300,400):#400]:#,500]:
#                freq = [123]
#            elif k in [200,600]:
###            print freq
            self.update(freq)
            
        return giMem,geMem,vMem

if __name__=='__main__':
    import pylab
    uwe = Test(nSteps=1000)
    gi,ge,v = uwe.start()
    figure = pylab.figure()
    vAx = pylab.subplot(311)
    pylab.plot(v)
    vAx.set_title('v')
    pylab.plot([0, 1000],[uwe._threshold, uwe._threshold])
    pylab.plot([0, 1000],[uwe._Ei, uwe._Ei])
    geAx = pylab.subplot(312)
    pylab.plot(ge)
    geAx.set_title('ge')
    giAx = pylab.subplot(313)
    pylab.plot(gi)
    giAx.set_title('gi')
    
    
    pylab.show()
'''

# initialize the neuron
neuron = DestexheNeuron()
