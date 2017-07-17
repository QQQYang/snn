# -----------complete XOR task------------
# author: qy
# 2017/7/5: the voltage of hidden layer declines to negative infinity, the updating formula needs to be check.
# 2017/7/7: 1.the issue above has been solved. What bothers me is that spiking neuron has too many parameters. How can I simplify the model and 
#           determine which parameters should be trained.
#           2. with training times increasing, 1 0 and 0 1 and 1 1 tend to output same spikes (nearly 24 pulses). Maybe STDP rules is pure feedforward.
#           So the weights keep increasing. How to limit the weights reasonably
#           3. mainly adjusted parameters: oneCycle, learning rate
# 2017/7/16:1.record every spike pair, and disable the spikes that have been used to update weights
import mylif
import stdp
import spikeGen
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.linalg as LA

#--------------------
# parameters
#-------------------
sampleNum = 80
t_total = 1
dt = 0.01
stepNum = int(t_total/dt)
zeroCycle = stepNum + 1
oneCycle = 2
cycle = (zeroCycle, oneCycle)
spikeSeq = np.zeros((2, stepNum))

#-------------------
# structure of network
#-------------------
numOfHidden = 3
numOfOutput = 1

#-------------------
# initialize weight randomly
# wih: w, weight; i, input; h, hidden
# who: w, weight; o, output
#-------------------
wih = np.random.rand(2,numOfHidden)
who = np.random.rand(numOfHidden,numOfOutput)
# assign weights equally
#wih = np.ones((2,numOfHidden))*1./(2*numOfHidden)
#who = np.ones((numOfHidden,numOfOutput))*1./(numOfHidden*numOfOutput)
# read weights from txt
#wih = np.loadtxt('wih.txt')
#who = np.loadtxt('who.txt')
# record previous weights
preWih = np.zeros((2, numOfHidden))
preWho = np.zeros((numOfHidden, numOfOutput))

#----------------------
# create netwotk
#----------------------
myNeuralParam = mylif.neuralParam(2, 0, 0.3, -68, 1, 0.03, -50, -70)
myStdpParam = stdp.stdpParam(0.01, 100, 0.01, 1000, 15, 1)
layer_h = mylif.layerModel(0, -68, numOfHidden)
layer_out = mylif.layerModel(0, -68, numOfOutput)

mode = 'train'

plt.ion()   # open interactive mode

if mode == 'train':
    for i in range(sampleNum*4):
        # start--------encode input 0,1-----------
        spikeSeq[0, :]=spikeGen.possionSpike(int(cycle[(i%4)>1]), stepNum)
        spikeSeq[1, :]=spikeGen.possionSpike(int(cycle[i%2]), stepNum)
        # end----------encode input 0,1------

        hSeq = np.zeros((numOfHidden, stepNum))
        hv = np.zeros((numOfHidden, stepNum))
        outSeq = np.zeros(stepNum)

        # start--------train network-------------
        mask_ih = np.zeros((2, numOfHidden))
        pair_ih = [[(x,y) for y in layer_h.spikeTime] for x in np.ones(2) * (-1) ]

        mask_ho = np.zeros((numOfHidden, numOfOutput))
        pair_ho = [[(x,y) for y in layer_out.spikeTime] for x in layer_h.spikeTime ]      
        for k in range(stepNum):
            hv[:, k] = layer_h.v
            # input -- first hidden layer
            spike_in_time = np.ones(2) * (-1)   # record the spiking time in layer1
            spike_in_time[spikeSeq[:, k] > 0] = k
            spike_h = layer_h.update(myNeuralParam, wih, spikeSeq[:, k], dt, k) # update the state of layer2
            dwih = stdp.stdp(wih, myStdpParam, spike_in_time, layer_h.spikeTime, 'bi') # calculate weight increase

            # record the spiking time pair between two layers
            #pair_ih[dwih == 0] = (-1,-1)
            ind = np.nonzero(dwih)
            for m in range(2):
                for n in range(numOfHidden):
                    if dwih[m,n] != 0:
                        if pair_ih[m][n][0] != spike_in_time[m] and pair_ih[m][n][1] != layer_h.spikeTime[n]:
                            pair_ih[m][n] = (spike_in_time[m], layer_h.spikeTime[n])
                            mask_ih[m][n] = 0

            dwih[mask_ih!=0] = 0
            wih += dwih
            # update mask
            mask_ih[dwih!=0] = 1

            # hidden layer -- output layer
            spike_out = layer_out.update(myNeuralParam, who, spike_h, dt, k)
            dwho = stdp.stdp(who, myStdpParam, layer_h.spikeTime, layer_out.spikeTime, 'bi')
            #who += dwho

            # record the spiking time pair between two layers
            #pair_ho[dwih == 0] = (-1,-1)
            ind = np.nonzero(dwho)
            for m in range(numOfHidden):
                for n in range(numOfOutput):
                    if dwho[m,n] != 0:
                        if pair_ho[m][n][0] != layer_h.spikeTime[m] and pair_ho[m][n][1] != layer_out.spikeTime[n]:
                            pair_ho[m][n] = (layer_h.spikeTime[m] , layer_out.spikeTime[n])
                            mask_ho[m][n] = 0

            dwho[mask_ho!=0] = 0
            who += dwho
            # update mask
            mask_ho[dwho!=0] = 1         

            # reset spiking time that has been used....to be discussed
            #layer_h.spikeTime = np.ones(numOfHidden) * (-1)
            #layer_out.spikeTime = np.ones(numOfOutput) * (-1)

            hSeq[:,k] = spike_h
            outSeq[k] = spike_out[0]

        # end--------train network-----------------
        
        # start------reset----------------
        layer_h.reset(0, -68)
        layer_out.reset(0, -68)
        # end--------reset----------------

        # start------judge convergence------
        if LA.norm(wih-preWih, "fro")<1e-6 and LA.norm(who-preWho, "fro")<1e-6:
            print('iteration: %d' % i)
            if i%4 != 0:
                break
        preWih = wih.copy()
        preWho = who.copy()
        # end--------judge convergence------

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

        #plt.title('the '+str(i)+ ' iteration ' + str(sum(outSeq)))
        #plt.plot(outSeq, 'b')
        #
        #plt.show() 
        #plt.waitforbuttonpress()
        #plt.clf()

    # start---------save weights--------
    np.savetxt('wih.txt', wih)
    np.savetxt('who.txt', who)
    # end-----------save weights--------
else:
    wih = np.loadtxt('wih.txt')
    who = np.loadtxt('who.txt')
    
    for i in range(4):
        spikeSeq[0, :]=spikeGen.possionSpike(int(cycle[(i%4)>1]), stepNum)
        spikeSeq[1, :]=spikeGen.possionSpike(int(cycle[i%2]), stepNum)

        outSeq = np.zeros(stepNum)

        for k in range(stepNum):
            spike_in_time = np.ones(2) * (-1)
            spike_in_time[spikeSeq[:, k] > 0] = k
            spike_h = layer_h.update(myNeuralParam, wih, spikeSeq[:, k], dt, k)

            spike_out = layer_out.update(myNeuralParam, who, spike_h, dt, k)
            outSeq[k] = spike_out[0]

        plt.title('the '+str(i)+ ' iteration ' + str(sum(outSeq)))
        plt.plot(outSeq, 'b')

        plt.show() 
        plt.waitforbuttonpress()
        plt.clf()