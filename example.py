import mylif
import stdp
import spikeGen
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt

def get_labeled_data(sampleFilename, labelFilename):
    samples = open(sampleFilename, 'rb')
    labels = open(labelFilename, 'rb')
    # Get metadata for images
    samples.read(4)  # skip the magic_number
    number_of_samples = unpack('>I', samples.read(4))[0]
    rows = unpack('>I', samples.read(4))[0]
    cols = unpack('>I', samples.read(4))[0]
    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = unpack('>I', labels.read(4))[0]

    if number_of_samples != N:
        raise Exception('number of labels did not match the number of samples')
    # Get the data
    x = np.zeros((N, rows, cols))#, dtype=np.uint8)  # Initialize numpy array
    y = np.zeros((N, 1))#, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        # if i % 1000 == 0:
        #   print("i: %i" % i)
        x[i] = [[unpack('>B', samples.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
        y[i] = unpack('>B', labels.read(1))[0]

    #np.savetxt("data/trainData.txt", x)
    #np.savetxt("data/trainLabel.txt", y)
    data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
    return data 

def save_weights(filename, w):
    print('save trained weights')   
    np.savetxt(filename, w)
#---------------------
# structure of network
#---------------------
numOfSpkieGenerate = 28*28
numOfHidden1 = 100
numOfHidden2 = 50
numOfHidden3 = 100
numOfOutput = 10

#-------------------
# parameters
#-------------------



#-------------------------
# random initialize weight with uniform distribution
#-------------------------
wi1 = np.random.rand(numOfSpkieGenerate, numOfHidden1)
w12 = np.random.rand(numOfHidden1, numOfHidden2)
w23 = np.random.rand(numOfHidden2, numOfHidden3)
w3o = np.random.rand(numOfHidden3, numOfOutput)

#---------------------------
# create networks
#---------------------------
myNeuralParam = mylif.neuralParam(2, 0, 0.3, -68, 1, 0.03, -50, -70)
# layer_in = mylif.layerModel(0, -68, numOfSpkieGenerate)
layer_h1 = mylif.layerModel(0, -68, numOfHidden1)
layer_h2 = mylif.layerModel(0, -68, numOfHidden2)
layer_h3 = mylif.layerModel(0, -68, numOfHidden3)
layer_out = mylif.layerModel(0, -68, numOfOutput)

myNeuralParam = mylif.neuralParam(2, 0, 0.3, -68, 1, 0.1, -50, -70)
myStdpParam = stdp.stdpParam(0.01, 100, 0.01, 1000, 15, 1)
#--------------------------
# stimulation
#--------------------------
t_total = 1
dt = 0.01
stepNum = int(t_total/dt)
mode = 'test'


if mode == 'train':
    #------------------
    # load MNIST
    #------------------
    trainDataName = 'data/train-images.idx3-ubyte'
    trainLabelName = 'data/train-labels.idx1-ubyte'
    trainData = get_labeled_data(trainDataName, trainLabelName)

    sampleNum = trainData['y'].size
    featureLength = trainData['x'].shape[1] * trainData['x'].shape[2]
    #-------------------------
    # extract subset
    #-------------------------
    subset = np.loadtxt('data/subset.txt', dtype=np.int, delimiter="\n")

    for i in subset:
        singleData = (trainData['x'][i-1, :, :].reshape((featureLength))).copy()
        singleData[singleData < 1] = 4./(dt*stepNum)
        spikeCycle = 4. / singleData.reshape((featureLength)) / dt
        spikeSeq = np.zeros((featureLength, stepNum))
        # generate input spike
        for j in range(featureLength):
            spikeSeq[j, :] = spikeGen.possionSpike(int(spikeCycle[j]), stepNum)

        # training
        mask_i1 = np.zeros((numOfSpkieGenerate, numOfHidden1))
        pair_i1 = [[(x,y) for y in layer_h1.spikeTime] for x in np.ones(numOfSpkieGenerate) * (-1) ]

        mask_12 = np.zeros((numOfHidden1, numOfHidden2))
        pair_12 = [[(x,y) for y in layer_h2.spikeTime] for x in layer_h1.spikeTime ]

        mask_23 = np.zeros((numOfHidden2, numOfHidden3))
        pair_23 = [[(x,y) for y in layer_h3.spikeTime] for x in layer_h2.spikeTime ]

        mask_3o = np.zeros((numOfHidden3, numOfOutput))
        pair_3o = [[(x,y) for y in layer_out.spikeTime] for x in layer_h3.spikeTime ]            

        for k in range(stepNum):
            # input layer and hidden layer 1
            spike_in_time = np.ones(featureLength) * (-1)
            spike_in_time[spikeSeq[:, k] > 0] = k
            spike_h1 = layer_h1.update(myNeuralParam, wi1, spikeSeq[:, k], dt, k)
            dwi1 = stdp.stdp(wi1, myStdpParam, spike_in_time, layer_h1.spikeTime, 'bi')
            wi1 += dwi1

            # record the spiking time pair between two layers
            for m in range(numOfSpkieGenerate):
                for n in range(numOfHidden1):
                    if dwi1[m,n] != 0:
                        if pair_i1[m][n][0] != spike_in_time[m] and pair_i1[m][n][1] != layer_h1.spikeTime[n]:
                            pair_i1[m][n] = (spike_in_time[m], layer_h1.spikeTime[n])
                            mask_i1[m][n] = 0

            dwi1[mask_i1!=0] = 0
            wi1 += dwi1
            # update mask
            mask_i1[dwi1!=0] = 1        

            # hidden layer 1 and hidden layer 2
            spike_h2 = layer_h2.update(myNeuralParam, w12, spike_h1, dt, k)
            dw12 = stdp.stdp(w12, myStdpParam, layer_h1.spikeTime, layer_h2.spikeTime, 'bi')
            w12 += dw12

            # record the spiking time pair between two layers
            for m in range(numOfHidden1):
                for n in range(numOfHidden2):
                    if dw12[m,n] != 0:
                        if pair_12[m][n][0] != layer_h1.spikeTime[m] and pair_12[m][n][1] != layer_h2.spikeTime[n]:
                            pair_12[m][n] = (layer_h1.spikeTime[m], layer_h2.spikeTime[n])
                            mask_12[m][n] = 0

            dw12[mask_12!=0] = 0
            w12 += dw12
            # update mask
            mask_12[dw12!=0] = 1        

            # hidden layer 2 and hidden layer 3
            spike_h3 = layer_h3.update(myNeuralParam, w23, spike_h2, dt, k)
            dw23 = stdp.stdp(w23, myStdpParam, layer_h2.spikeTime, layer_h3.spikeTime, 'bi')
            w23 += dw23

            # record the spiking time pair between two layers
            for m in range(numOfHidden2):
                for n in range(numOfHidden3):
                    if dw23[m,n] != 0:
                        if pair_23[m][n][0] != layer_h2.spikeTime[m] and pair_23[m][n][1] != layer_h3.spikeTime[n]:
                            pair_23[m][n] = (layer_h2.spikeTime[m], layer_h3.spikeTime[n])
                            mask_23[m][n] = 0

            dw23[mask_23!=0] = 0
            w23 += dw23
            # update mask
            mask_23[dw23!=0] = 1        

            # hidden layer 3 and output layer
            spike_out = layer_out.update(myNeuralParam, w3o, spike_h3, dt, k)
            dw3o = stdp.stdp(w3o, myStdpParam, layer_h3.spikeTime, layer_out.spikeTime, 'bi')
            w3o += dw3o

            # record the spiking time pair between two layers
            for m in range(numOfHidden3):
                for n in range(numOfOutput):
                    if dw3o[m,n] != 0:
                        if pair_3o[m][n][0] != layer_h3.spikeTime[m] and pair_3o[m][n][1] != layer_out.spikeTime[n]:
                            pair_3o[m][n] = (layer_h3.spikeTime[m], layer_out.spikeTime[n])
                            mask_3o[m][n] = 0

            dw3o[mask_3o!=0] = 0
            w3o += dw3o
            # update mask
            mask_3o[dw3o!=0] = 1        

        # reset
        layer_h1.reset(0, -68)
        layer_h2.reset(0, -68)
        layer_h3.reset(0, -68)
        layer_out.reset(0, -68)

    # save trained weights
    save_weights('wi1.txt', wi1)
    save_weights('w12.txt', w12)
    save_weights('w23.txt', w23)
    save_weights('w3o.txt', w3o)
else:
    # load MNIST test dataset
    testDataName = 'data/t10k-images.idx3-ubyte'
    testLabelName = 'data/t10k-labels.idx1-ubyte'
    testData = get_labeled_data(testDataName, testLabelName)
    featureLength = testData['x'].shape[1] * testData['x'].shape[2]

    # load weights from txt
    wi1 = np.loadtxt('weights/wi1.txt')/5.
    w12 = np.loadtxt('weights/w12.txt')/5.
    w23 = np.loadtxt('weights/w23.txt')/5.
    w3o = np.loadtxt('weights/w3o.txt')/5.

    # extract subset
    subset = np.loadtxt('weights/subset.txt', dtype=np.int, delimiter="\n")

    count = 0

    # start test
    for i in subset:
        count += 1
        singleData = (testData['x'][i-1, :, :].reshape((featureLength))).copy()
        singleData[singleData < 1] = 4./(dt*stepNum)
        spikeCycle = 4. / singleData.reshape((featureLength)) / dt
        spikeSeq = np.zeros((featureLength, stepNum))
        # generate input spike
        for j in range(featureLength):
            spikeSeq[j, :] = spikeGen.possionSpike(int(spikeCycle[j]), stepNum)  

        outSeq = np.zeros((stepNum, numOfOutput))

        for k in range(stepNum):
            # input layer and hidden layer 1
            spike_in_time = np.ones(featureLength) * (-1)
            spike_in_time[spikeSeq[:, k] > 0] = k
            spike_h1 = layer_h1.update(myNeuralParam, wi1, spikeSeq[:, k], dt, k)    

            # hidden layer 1 and hidden layer 2
            spike_h2 = layer_h2.update(myNeuralParam, w12, spike_h1, dt, k)   

            # hidden layer 2 and hidden layer 3
            spike_h3 = layer_h3.update(myNeuralParam, w23, spike_h2, dt, k)           

            # hidden layer 3 and output layer
            spike_out = layer_out.update(myNeuralParam, w3o, spike_h3, dt, k)

            outSeq[k, :] = spike_out.copy()

        # draw results
        #plt.subplots(nrows=2, ncols=5)
        #for j in range(10):
        #    plt.subplot(2, 5, j+1)
        #    plt.title('the '+str(count)+ ' iteration ' + str(sum(outSeq[:,j])))
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.clf()

        # select the neuron that spikes the most
        maxSpike = 0
        maxInd = 1
        for j in range(10):
            if sum(outSeq[:,j]) > maxSpike:
                maxSpike = sum(outSeq[:,j])
                maxInd = j
        print('spikes:%d, class:%d, label:%d\n' % (maxSpike, maxInd+1, count))

        layer_h1.reset(0, -68)
        layer_h2.reset(0, -68)
        layer_h3.reset(0, -68)
        layer_out.reset(0, -68)        
        #plt.title('the '+str(count)+ ' iteration, ' + 'spikes: ' + str(maxSpike) + ', label: ' + str(maxInd))
        #plt.plot(outSeq[:,maxInd], 'b')
#
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.clf()        