import numpy as np
import matplotlib.pyplot as plt

#neural parameters
class neuralParam(object):

    def __init__(self, tau_e, v_e_syn, g_l, v_l, c_m, t_ref, v_thr, v_res):
        self.tau_e = tau_e  # ms
        self.v_e_syn = v_e_syn  # mV
        self.g_l = g_l  # mS/cm^2
        self.v_l = v_l  # mV
        self.c_m = c_m  # uF/cm^2
        self.t_ref = t_ref  # ms
        self.v_thr = v_thr  # mV
        self.v_res = v_res  # mV

#single neuron model
class neuralModel(object):

    def __init__(self, g_e, v):
        self.g_e = g_e
        self.v = v
        self.t = 0
        self.spike = 0

    def update(self, param, w, sp, dt):
        a_e = (2*param.tau_e - dt)/(2*param.tau_e + dt)
        b_e = 2/(2*param.tau_e + dt)
        self.g_e = a_e * self.g_e + b_e * np.dot(w, sp)
        top = (2*param.c_m/dt - (param.g_l + self.g_e))*self.v + 2*param.g_l*param.v_l
        bot = 2*param.c_m/dt + param.g_l + self.g_e

        # refractory
        if self.t > 0:
            self.v = param.v_res
        else:
            self.v = top/bot

        # threhold judgement
        if self.v > param.v_thr:
            self.v = param.v_res
            self.spike = 1
            self.t = int(param.t_ref/dt)
        else:
            self.spike = 0
            if self.t > 0:
                self.t -= 1

        return self.spike

#model of every layer of the whole network
#replace single state with vector
class layerModel(object):

    def __init__(self, g_e, v, num):
        self.g_e = np.ones(num) * g_e
        self.v = np.ones(num) * v
        self.t = np.zeros(num)
        self.spike = np.zeros(num)
        self.num = num
        self.spikeTime = np.ones(num) * (-1)

    def update(self, param, w, sp, dt, curTime):
        a_e = (2*param.tau_e - dt)/(2*param.tau_e + dt)
        b_e = 2/(2*param.tau_e + dt)
        g_e_pre = self.g_e
        self.g_e = np.dot(a_e, self.g_e) + np.dot(b_e, np.dot(w.T, sp)) #weight has effect on conductance
        top = (np.dot(np.ones(self.num), (2*param.c_m/dt)) - (np.dot(np.ones(self.num), param.g_l) + g_e_pre))*self.v + 2*param.g_l*param.v_l*np.ones(self.num) \
           + (self.g_e + g_e_pre)*param.v_e_syn
        bot = (2*param.c_m/dt + param.g_l) * np.ones(self.num) + self.g_e

        # refractory
        if abs(bot[0])<1e-6:
            print('oh! my god! What happened')
        self.v = top/bot
        self.v[self.v < param.v_res] = param.v_res  # limit the voltage
        self.v[self.t > 0] = param.v_res

        # threhold judgement
        spikeInd = np.nonzero(self.v > param.v_thr)
        self.spike[spikeInd] = 1
        self.spikeTime[spikeInd] = curTime  # record spike time
        #self.t[spikeInd] = int(param.t_ref/dt)

        noSpikeInd = np.nonzero(self.v <= param.v_thr)
        self.v[spikeInd] = param.v_res
        self.spike[noSpikeInd] = 0
        #self.t[np.nonzero(np.logical_and(self.v <= param.v_thr, self.t > 0))] -= 1
        self.t[self.t > 0] -= 1
        self.t[spikeInd] = int(param.t_ref/dt)

        return self.spike

    def reset(self, g_e, v):
        self.g_e = np.ones(self.num) * g_e
        self.v = np.ones(self.num) * v
        self.t = np.zeros(self.num)
        self.spike = np.zeros(self.num)
        self.spikeTime = np.ones(self.num) * (-1)

if 0:
    t_total = 50
    dt = 0.01
    wi1 = [[0.5, 0.5], [0.4, 0.6]]
    w12 = [[0.8, 0.5], [0.2, 0.5]]
    w13 = 0.5
    w23 = 0.5

    stepNum = int(t_total/dt)
    neuronNumi1 = int(2)
    neuronNum12 = int(2)

    gei1 = np.zeros((neuronNumi1, stepNum))
    vi1 = np.zeros((neuronNumi1, stepNum))
    ge12 = np.zeros((neuronNum12, stepNum))
    v12 = np.zeros((neuronNum12, stepNum))
    spi1 = np.zeros(neuronNumi1)
    sp12 = np.zeros(neuronNum12)
    t = range(stepNum)

    myNeuralParam = neuralParam(2, 0, 0.3, -68, 1, 3, -50, -70)
    neuralNetworkLayeri1 = []
    neuralNetworkLayer12 = []
    for j in range(neuronNumi1):
        singleNeuralModeli1 = neuralModel(0, -68)
        neuralNetworkLayeri1.append(neuralModel(0, -68))

    for j in range(neuronNum12):
        singleNeuralModel12 = neuralModel(0, -68)
        neuralNetworkLayer12.append(neuralModel(0, -68))

    for i in range(stepNum):
        if i%500 == 499:
            spi = 1
        else:
            spi = 0

        for j in range(neuronNumi1):
            spi1[j] = neuralNetworkLayeri1[j].update(myNeuralParam, wi1[:, j], [spi, spi], dt)
            gei1[j][i] = neuralNetworkLayeri1[j].g_e
            vi1[j][i] = neuralNetworkLayeri1[j].v

        for j in range(neuronNum12):
            neuralNetworkLayer12[j].update(myNeuralParam, w12[:, j], spi1, dt)
            ge12[j][i] = neuralNetworkLayer12[j].g_e
            v12[j][i] = neuralNetworkLayer12[j].v

    # plot
    fig, ax = plt.subplots(4, 2)
    ax[0][0].plot(t, gei1[0])
    ax[1][0].plot(t, gei1[1])
    ax[2][0].plot(t, ge12[0])
    ax[3][0].plot(t, ge12[1])
    ax[0][1].plot(t, vi1[0])
    ax[1][1].plot(t, vi1[1])
    ax[2][1].plot(t, v12[0])
    ax[3][1].plot(t, v12[1])
    plt.show()