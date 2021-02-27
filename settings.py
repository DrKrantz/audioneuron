# neuron specific parameters
neuronId = 6
neuronalFrequency = 880
neuronalType = 'I'
presynapticFrequencies = 1760, 1661, 959, 1863, 1209
presynapticNeurons = {0: 'E', 1: 'E', 2: 'I', 3: 'E', 4: 'E'}


# Generic sound parameters
frequencyThreshold = 150
channel = 'LR'  # or 'R' or 'LR'
toneDuration = 500  # ms
frequencyTolerance = 0.1  # in fractions of half-notes
dampDuration = 0.01  # time for linear in/decrease of sine, in seconds
endSilence = 0.005  # time for silence after the sine, in seconds
NOTERATIO = 2**(1./12)

# Display parameters
displaySize = (350, 700)
fftDisplayXLimits = [300, 5000]
colors = {'E': [0, 0, 255], 'I': [255, 255, 0], 'fft': [100, 100, 100], 'membrane': [100, 100, 100]}


def defaultPars(neuron_type='E'):
    pars = dict()
    # parameters of the neuron model
    pars['threshold'] = -50e-3  # firing threshold of individual neurons, V
    pars['Cm'] = 1e-6  # membrane capacitance, F/cm2
    pars['gL'] = 0.05e-3  # leak conductances, S/cm2
    pars['EL'] = -60e-3  # resting potential = reset after spike, V
    pars['Delta'] = 2.5e-3  # steepness of exponential, V
    pars['S'] = 20000e-8  # membrane area, cm2
    pars['dead'] = 1e-2  # 2.5e-3 # deadtime
    
    # type=dependent parameters
    a_e = 0  # 0.08e-6 # adaptation dynamics of e-synapses, S
    a_i = 0
    b_e = 0
    b_i = 0

    pars['a'] = a_e if neuron_type == 'E' else a_i
    pars['b'] = b_e if neuron_type == 'E' else b_i
    
    # Parameters of Synapses 
    pars['s_e'] = 6e-9  # 10increment of excitatory synaptic conductance per spike, S,
    pars['s_i'] = 67e-9  # increment of inhibitory synaptic conductance S per spike
    pars['tau_e'] = 5e-3  # sec,
    pars['tau_i'] = 10e-3  # sec
    pars['tau_w'] = 600e-3  # time-constant of adaptation variable, sec
    pars['Ee'] = 0e-3  # reversal potential of excitatory synapses, V
    pars['Ei'] = -80e-3  # reversal potential of inhibitory synapses, V
    
    pars['external'] = 1e-7

    return pars
