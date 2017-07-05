import scipy.stats as scistats
import matplotlib.pyplot as plt
import numpy as np

#cycle = (1/freq)/dt
def possionSpike(cycle, length):
    if cycle>length:
        return np.zeros(length)
    interval = scistats.poisson.rvs(mu=cycle, size=length)
    spiketime = interval.cumsum()
    stimus = np.zeros(length)
    stimus[0] = 1
    stimus[spiketime[spiketime < length]] = 1
    return stimus

def copyNoiseSpike(image, length, grayThres):
    # 2-D to 1-D
    imageArray = image.flatten(1)
    imageArray[imageArray > grayThres] = 1

    # add Poisson noise
    # how to add Poisson noise to binary image
    stimus = np.dot(imageArray.reshape((-1, 1)), np.ones((1, length)))
    imageSize = imageArray.size
    return stimus

