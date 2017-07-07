# -----------complete XOR task------------
# author: qy
# 2017/7/5: the voltage of hidden layer declines to negative infinity, the updating formula needs to be check.
# 2017/7/7: the issue above has been solved. What bothers me is that spiking neuron has too many parameters. How can I simplify the model and 
#           determine which parameters should be trained.
import mylif
import stdp
import spikeGen
import numpy as np
import matplotlib.pyplot as plt
import time

#--------------------
# parameters
#-------------------
sampleNum = 10
t_total = 1
dt = 0.01
stepNum = int(t_total/dt)
zeroCycle = stepNum + 1
oneCycle = 5
cycle = (zeroCycle, oneCycle)
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
wih = np.ones((2,numOfHidden))*1./(2*numOfHidden)
who = np.ones((numOfHidden,numOfOutput))*1./(numOfHidden*numOfOutput)

#----------------------
# create netwotk
#----------------------
myNeuralParam = mylif.neuralParam(2, 0, 0.3, -68, 1, 0.03, -50, -70)
myStdpParam = stdp.stdpParam(0.1, 100, 0.1, 1000, 15, 1)
layer_h = mylif.layerModel(0, -68, numOfHidden)
layer_out = mylif.layerModel(0, -68, numOfOutput)

mode = 'test'

plt.ion()   # open interactive mode

for i in range(sampleNum*4):
    # start--------encode input 0,1-----------
    spikeSeq[0, :]=spikeGen.possionSpike(int(cycle[(i%4)>1]), stepNum)
    spikeSeq[1, :]=spikeGen.possionSpike(int(cycle[i%2]), stepNum)
    # end----------encode input 0,1------

    hSeq = np.zeros((numOfHidden, stepNum))
    hv = np.zeros((numOfHidden, stepNum))
    outSeq = np.zeros(stepNum)

    # start--------train network-------------
    for k in range(stepNum):
        hv[:, k] = layer_h.v
        # input -- first hidden layer
        spike_in_time = np.ones(2) * (-1)   # record the spiking time in layer1
        spike_in_time[spikeSeq[:, k] > 0] = k
        spike_h = layer_h.update(myNeuralParam, wih, spikeSeq[:, k], dt, k) # update the state of layer2
        dwih = stdp.stdp(wih, myStdpParam, spike_in_time, layer_h.spikeTime, 'bi') # calculate weight increase
        wih += dwih

        # hidden layer -- output layer
        spike_out = layer_out.update(myNeuralParam, who, spike_h, dt, k)
        dwho = stdp.stdp(who, myStdpParam, layer_h.spikeTime, layer_out.spikeTime, 'bi')
        who += dwho

        # reset spiking time that has been used....to be discussed
        layer_h.spikeTime = np.ones(numOfHidden) * (-1)
        layer_out.spikeTime = np.ones(numOfOutput) * (-1)

        hSeq[:,k] = spike_h
        outSeq[k] = spike_out[0]

    # end--------train network-----------------

    # start------reset----------------
    layer_h.reset(0, -68)
    layer_out.reset(0, -68)
    # end--------reset----------------

    #plt.figure(1)
    #plt.subplot(321)
    #plt.title('the '+str(i)+ ' iteration')
    #plt.plot(hSeq[0,:], 'b')
    #plt.subplot(322)
    #plt.plot(hv[0,:], 'b')

    #plt.subplot(323)
    #plt.plot(hSeq[1,:], 'g')
    #plt.subplot(324)
    #plt.plot(hv[1,:], 'g')

    #plt.subplot(325)
    #plt.plot(hSeq[2,:], 'r')
    #plt.subplot(326)
    #plt.plot(hv[2,:], 'r')

    plt.title('the '+str(i)+ ' iteration ' + str(sum(outSeq)))
    plt.plot(outSeq, 'b')
    
    plt.show() 
    plt.waitforbuttonpress()
    plt.clf()

# start---------save weights--------
#np.savetxt('wih.txt', wih)
#np.savetxt('who.txt', who)
# end-----------save weights--------