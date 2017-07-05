# -----------complete XOR task------------
# author: qy
import mylif
import stdp
import spikeGen
import numpy as np

#--------------------
# parameters
#-------------------
sampleNum = 10
t_total = 1
dt = 0.01
stepNum = int(t_total/dt)
zeroCycle = stepNum + 1
oneCycle = 5
cycle = (zeroCycle oneCycle)
spikeSeq = np.zeros((2, stepNum))

#-------------------
# structure of network
#-------------------
numOfHidden = 3
numOfOutput = 1

#-------------------
# initialize weight randomly
#-------------------
wih = np.random.rand(2,numOfHidden)
who = np.random.rand(numOfHidden,numOfOutput)

#----------------------
# create netwotk
#----------------------
myNeuralParam = mylif.neuralParam(2, 0, 0.3, -68, 1, 3, -50, -70)
myStdpParam = stdp.stdpParam(0.3, 100, 0.3, 1000, 15, 1)
layer_h = mylif.layerModel(0, -68, numOfHidden)
layer_out = mylif.layerModel(0, -68, numOfOutput)


for i in range(sampleNum*4):
    # start--------encode input 0,1-----------
    spikeSeq[0, :]=spikeGen.possionSpike(int(cycle[(i%4)>1]), stepNum)
    spikeSeq[1, :]=spikeGen.possionSpike(int(cycle[i%2]), stepNum)
    # end----------encode input 0,1------

    # start--------train network-------------
    for k in range(stepNum):
        # input -- first hidden layer
        spike_in_time = np.ones(2) * (-1)   # record the spiking time in layer1
        spike_in_time[spikeSeq[:, k] > 0] = k
        spike_h = layer_h.update(myNeuralParam, wih, spikeSeq[:, k], dt, k) # update the state of layer2
        dwih = stdp.stdp(wih, myStdpParam, spike_in_time, layer_h.spikeTime, 'tri') # calculate weight increase
        wih += dwih

        # hidden layer -- output layer
        layer_out.update(myNeuralParam, who, spike_h, dt, k)
        dwho = stdp.stdp(who, myStdpParam, layer_h.spikeTime, layer_out.spikeTime, 'tri')
        who += dwho