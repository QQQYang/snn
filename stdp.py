# STDP learning rules

import numpy as np
from math import e

class stdpParam(object):

    def __init__(self, A_pos, tau_pos, A_neg, tau_neg, t_max, w_max):
        self.A_pos = A_pos
        self.tau_pos = tau_pos
        self.A_neg = A_neg
        self.tau_neg = tau_neg
        self.t_max = t_max
        self.w_max = w_max

def stdp(w, param, spiketimePre, spiketimePost, stdpType):
    preNum = spiketimePre.size
    postNum = spiketimePost.size
    # column copy to matrix with size of preNum*postNum
    spiketimePreCopy = np.dot(spiketimePre.reshape((-1, 1)), np.ones((1, postNum)))
    # row copy to matrix with size of preNum*postNum
    spiketimePostCopy = np.dot(np.ones((preNum, 1)), spiketimePost.reshape((1, -1)))

    spiketimeDelta = spiketimePreCopy - spiketimePostCopy

    if stdpType == 'tri':
        # tri-phasic STDP
        dw = param.A_pos * e**(-(spiketimeDelta**2)/param.tau_pos) - param.A_neg * e**(-(spiketimeDelta**2)/param.tau_neg)
        dw[spiketimePostCopy < 0] = 0
        dw[spiketimePreCopy < 0] = 0
        return dw
    else:
        # bi-phasic STDP
        learnRate = np.ones((preNum, postNum)) * param.A_pos
        learnRate[spiketimeDelta > 0] = param.A_neg
        spiketimeDelta[spiketimeDelta > 0] = -spiketimeDelta[spiketimeDelta > 0]
        tau = np.ones((preNum, postNum)) * param.tau_pos
        tau[spiketimeDelta > 0] = param.tau_neg
        dw = learnRate * e**(spiketimeDelta/tau)
        dw[spiketimePostCopy < 0] = 0
        dw[spiketimePreCopy < 0] = 0
        return dw