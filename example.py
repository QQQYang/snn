import mylif
import stdp
import spikeGen
from struct import unpack
import numpy as np

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

#------------------
# load MNIST
#------------------
trainDataName = 'data/train-images.idx3-ubyte'
trainLabelName = 'data/train-labels.idx1-ubyte'
trainData = get_labeled_data(trainDataName, trainLabelName)

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
myNeuralParam = mylif.neuralParam(2, 0, 0.3, -68, 1, 3, -50, -70)
# layer_in = mylif.layerModel(0, -68, numOfSpkieGenerate)
layer_h1 = mylif.layerModel(0, -68, numOfHidden1)
layer_h2 = mylif.layerModel(0, -68, numOfHidden2)
layer_h3 = mylif.layerModel(0, -68, numOfHidden3)
layer_out = mylif.layerModel(0, -68, numOfOutput)

myNeuralParam = mylif.neuralParam(2, 0, 0.3, -68, 1, 3, -50, -70)
myStdpParam = stdp.stdpParam(0.3, 100, 0.3, 1000, 15, 1)
#--------------------------
# stimulation
#--------------------------
t_total = 1
dt = 0.01
stepNum = int(t_total/dt)

sampleNum = trainData['y'].size
featureLength = trainData['x'].shape[1] * trainData['x'].shape[2]

for i in range(sampleNum):
    singleData = (trainData['x'][i, :, :].reshape((featureLength))).copy()
    singleData[singleData < 1] = 4./(dt*stepNum)
    spikeCycle = 4. / singleData.reshape((featureLength)) / dt
    spikeSeq = np.zeros((featureLength, stepNum))
    # generate input spike
    for j in range(featureLength):
        spikeSeq[j, :] = spikeGen.possionSpike(int(spikeCycle[j]), stepNum)

    for k in range(stepNum):
        # input layer and hidden layer 1
        spike_in_time = np.ones(featureLength) * (-1)
        spike_in_time[spikeSeq[:, k] > 0] = k
        spike_h1 = layer_h1.update(myNeuralParam, wi1, spikeSeq[:, k], dt, k)
        dwi1 = stdp.stdp(wi1, myStdpParam, spike_in_time, layer_h1.spikeTime, 'tri')
        wi1 += dwi1

        # hidden layer 1 and hidden layer 2
        spike_h2 = layer_h2.update(myNeuralParam, w12, spike_h1, dt, k)
        dw12 = stdp.stdp(w12, myStdpParam, layer_h1.spikeTime, layer_h2.spikeTime, 'tri')
        w12 += dw12

        # hidden layer 2 and hidden layer 3
        spike_h3 = layer_h3.update(myNeuralParam, w23, spike_h2, dt, k)
        dw23 = stdp.stdp(w23, myStdpParam, layer_h2.spikeTime, layer_h3.spikeTime, 'tri')
        w23 += dw23

        # hidden layer 3 and output layer
        spike_out = layer_out.update(myNeuralParam, w3o, spike_h3, dt, k)
        dw3o = stdp.stdp(w3o, myStdpParam, layer_h3.spikeTime, layer_out.spikeTime, 'tri')
        w3o += dw3o

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