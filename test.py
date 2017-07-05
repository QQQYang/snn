from brian2 import *

#structure of network
numOfSpkieGenerate = 28*28
numOfHidden1 = 100
numOfHidden2 = 50
numOfHidden3 = 100
numOfOutput = 10

#parameters
tau_m = 20*ms
tau_e = 2*ms
tau_i = 5*ms
tau_pos = 100
tau_neg = 1000
t_max = 15

Vl = -70*mV
Vth = -50*mV
Vreset = -55*mV
Ve = 0*mV
Vi = -70*mV
gL = 20

#neural model
eqs = '''
dv/dt = (-(v - Vl)-gE/gL*(v-Ve)-gI/gL*(v-Vi))/tau_m : volt
gE : 1
gI : 1
'''

#construct each layer
inputLayer = NeuronGroup(numOfSpkieGenerate, eqs, threshold='v>Vth', reset='v=Vreset')
hiddenLayer1 = NeuronGroup(numOfHidden1, eqs, threshold='v>Vth', reset='v=Vreset')
hiddenLayer2 = NeuronGroup(numOfHidden2, eqs, threshold='v>Vth', reset='v=Vreset')
hiddenLayer3 = NeuronGroup(numOfHidden3, eqs, threshold='v>Vth', reset='v=Vreset')
outputLayer = NeuronGroup(numOfOutput, eqs, threshold='v>Vth', reset='v=Vreset')

#Inhibitory
exProportion = 0.75
hiddenLayer1Ex = hiddenLayer1[:numOfHidden1*exProportion]
hiddenLayer1In = hiddenLayer1[numOfHidden1*exProportion:]

hiddenLayer2Ex = hiddenLayer2[:numOfHidden2*exProportion]
hiddenLayer2In = hiddenLayer2[numOfHidden2*exProportion:]

hiddenLayer3Ex = hiddenLayer3[:numOfHidden3*exProportion]
hiddenLayer3In = hiddenLayer3[numOfHidden3*exProportion:]

outputLayerEx = outputLayer[:numOfOutput*exProportion]
outputLayerIn = outputLayer[numOfOutput*exProportion:]

##construct synapse
#input ---- hidden1
model_input_hidden1='''
pulse_input_hidden1 : Hz
w_input_hidden1 : Hz
dgE/dt = -gE/tau_e + pulse_input_hidden1: Hz
dgI/dt = -gI/tau_i + pulse_input_hidden1: Hz
'''
S_input_hidden1 = Synapses(inputLayer, hiddenLayer1, model=model_input_hidden1, on_pre='pulse_input_hidden1 += w_input_hidden1')

#hidden1 ---- hidden2
model_hidden12='''
pulse_hidden12_ex : Hz
pulse_hidden12_in : Hz
w_hidden12_ex : Hz
w_hidden12_in : Hz
dgE/dt = -gE/tau_e + pulse_hidden12_ex: Hz
dgI/dt = -gI/tau_i + pulse_hidden12_in: Hz
'''
S_hidden12_ex_ex = Synapses(hiddenLayer1Ex, hiddenLayer2Ex, model=model_hidden12, on_pre='pulse_hidden12_ex += w_hidden12_ex')
S_hidden12_in_in = Synapses(hiddenLayer1In, hiddenLayer2In, model=model_hidden12, on_pre='pulse_hidden12_in += w_hidden12_in')
