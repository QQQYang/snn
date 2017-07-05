'''
Interactive module for integrate-and-fire model.
John D. Murray (john.david.murray@gmail.com)
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib as mpl

dt = 0.05 # Integration  dt [ms]

# Make time vector for bottom axis
T = 750
t = np.arange(0,T,dt)-50
Iinj = np.zeros(len(t))
for i,ti in enumerate(t):
    if (0 < ti) & (ti < 500):
        Iinj[i] = 1

# F(I) curve; firing rate as a function of current and parameters
def f(I,A,R,C,vth,tref,vl,vres):
    if np.shape(I):
        z = np.zeros(len(I))
        ind = np.where(I>(vth-vl)/(R/A))[0]
        z[ind] = 1000.*(tref + R*C*np.log((R/A*I[ind] + vl - vres)/(R/A*I[ind] + vl - vth)))**-1
    else:
        if (I>(vth-vl)/(R/A)):
            z = 1000.*(I>(vth-vl)/(R/A))*(tref + R*C*np.log((R/A*I + vl - vres)/(R/A*I + vl - vth)))**-1
        else:
            z = 0
    return z

def dyn_fn(I, dt, A,R,C,vres,tref,vl,vth): # Integrates trajectory for neuron dynamics
    tlastspike = -100
    x = np.zeros(len(I))
    x[0] = vl
    for i in xrange(1,len(I)):
        if i*dt > tlastspike + tref+dt:
            x[i] = x[i-1] +dt*(-1./(R*C)*(x[i-1]-vl) + 1./(C*A)*I[i]) # Forward Euler
            if x[i] > vth:
                x[i]=0.
                tlastspike=i*dt
        else:
            x[i] = vres
    return x

# Set plotting properties

params = {'axes.labelsize': 14,
          'text.fontsize': 14,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10}
mpl.rcParams.update(params)
mpl.rc('mathtext', fontset='stixsans',default='regular')

# Make figure and axes

fig = plt.figure(figsize=(6,6))
fig.text(.5,.95,'Integrate-and-Fire model',ha='center',size=18)
ax = fig.add_axes([.1,.1,.85,.275]) # bottom axes, vm vs time
axi = fig.add_axes([.1,.4,.85,.075]) # middle axes, injected current vs time
ax1 = fig.add_axes([.1,.6,.4,.3]) # top axes, firing rate vs current

ax.set_xlabel('Time [ms]')
ax.set_ylabel('Potential [mV]')
ax.set_xlim(-50,700)
ax.set_ylim(-75,-40)
ax.set_yticks(np.arange(-70,-45,10))

axi.set_ylabel('I [nA]')
axi.set_xlim(-50,700)
axi.set_xticks([])
axi.set_yticks([0,1])
axi.set_ylim(0,1.05)

ax1.set_xlabel('Injected current [nA]')
ax1.set_ylabel('Firing rate [Hz]')
ax1.set_xlim(0,1)
ax1.set_ylim(0,100)

axcolor = 'lightgoldenrodyellow'
axs_I  = plt.axes([0.7, 0.85, 0.2, 0.03], axisbg=axcolor)
axs_A  = plt.axes([0.7, 0.5, 0.2, 0.03], axisbg=axcolor)
axs_R  = plt.axes([0.7, 0.6, 0.2, 0.03], axisbg=axcolor)
axs_C  = plt.axes([0.7, 0.55, 0.2, 0.03], axisbg=axcolor)
axs_vl  = plt.axes([0.7, 0.65, 0.2, 0.03], axisbg=axcolor)
axs_vth  = plt.axes([0.7, 0.8, 0.2, 0.03], axisbg=axcolor)
axs_vres  = plt.axes([0.7, 0.75, 0.2, 0.03], axisbg=axcolor)
axs_tref  = plt.axes([0.7, 0.7, 0.2, 0.03], axisbg=axcolor)

# Make sliders that control parameters

s_I = Slider(axs_I, r'$I \, [nA]$', 0, 1.0, valinit=0.5,color='maroon')
s_A = Slider(axs_A, r'$A \, [mm^2\!]$', 0.01, 0.2, valinit=0.05,color='midnightblue')
s_R = Slider(axs_R, r'$r \, [M\Omega\ mm^2\!]$', 0.1, 10.0, valinit=2,color='midnightblue')
s_C = Slider(axs_C, r'$c \, [nF/mm^2\!]$', 1., 20.0, valinit=10,color='midnightblue')
s_vl = Slider(axs_vl, r'$V_L \, [mV]$', -75, -45, valinit=-70,color='midnightblue')
s_vth = Slider(axs_vth, r'$V_{th} \, [mV]$', -52, -41, valinit=-52,color='maroon')
s_vres = Slider(axs_vres, r'$V_{res} \, [mV]$', -70, -53, valinit=-60,color='maroon')
s_tref = Slider(axs_tref, r'$\tau_{ref} \, [ms]$', 0, 10, valinit=2,color='maroon')

fs1=10
s_I.label.set_fontsize(fs1)
s_A.label.set_fontsize(fs1)
s_R.label.set_fontsize(fs1)
s_C.label.set_fontsize(fs1)
s_vl.label.set_fontsize(fs1)
s_vth.label.set_fontsize(fs1)
s_vres.label.set_fontsize(fs1)
s_tref.label.set_fontsize(fs1)

# Make plots

Is = np.linspace(0,10,1000)
lv = ax1.axvline(s_I.val,c='grey',alpha=0.5)
lh = ax1.axhline(f(s_I.val, A = s_A.val, R=s_R.val, C=s_C.val, vl=s_vl.val, vth = s_vth.val, vres=s_vres.val, tref=s_tref.val),c='grey',alpha=0.5)
l, = ax1.plot(Is,f(Is, A = s_A.val,R=s_R.val, C=s_C.val, vl=s_vl.val, vth = s_vth.val, vres=s_vres.val, tref=s_tref.val),c='r',lw=2)
sc = ax1.scatter([s_I.val],[f(s_I.val, A = s_A.val, R=s_R.val, C=s_C.val, vl=s_vl.val, vth = s_vth.val, vres=s_vres.val, tref=s_tref.val)],s=50,c='blue')

li, = axi.plot(t,s_I.val*Iinj,c='maroon')

lb, = ax.plot(t,dyn_fn(s_I.val*Iinj,dt,A = s_A.val, R=s_R.val, C=s_C.val, vl=s_vl.val, vth = s_vth.val, vres=s_vres.val, tref=s_tref.val),c='b')
lhb = ax.axhline(s_vth.val,ls='--',c='r')

# Update initial condition and parameters

def update(val):
    R=s_R.val
    C=s_C.val
    vl=s_vl.val
    vth = s_vth.val
    vres=s_vres.val
    tref=s_tref.val
    A = s_A.val
    x = s_I.val
    y = f(s_I.val, A=s_A.val,R=s_R.val, C=s_C.val, vl=s_vl.val, vth = s_vth.val, vres=s_vres.val, tref=s_tref.val)
    lh.set_ydata(y)
    lv.set_xdata(x)
    lhb.set_ydata(s_vth.val)
    l.set_ydata(f(Is,A=A, R=R, C=C, vl=vl, vth = vth, vres=vres, tref=tref))
    ax1.collections.pop()
    lb.set_ydata(dyn_fn(s_I.val*Iinj,dt,A=A,vl=vl,vth=vth,vres=vres,tref=tref,C=C,R=R))
    sc = ax1.scatter(x,y,s=50,c='blue')
    ax1.set_ylim(0,100)
    ax1.set_xlim(0,1)
    li.set_ydata(s_I.val*Iinj)
    plt.draw()

s_I.on_changed(update)
s_R.on_changed(update)
s_C.on_changed(update)
s_tref.on_changed(update)
s_vres.on_changed(update)
s_vl.on_changed(update)
s_vth.on_changed(update)
s_A.on_changed(update)


# Make reset button
resetax = plt.axes([0.8, 0.9, 0.1, 0.04]) # Reset button
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    s_I.reset()
    s_R.reset()
    s_C.reset()
    s_tref.reset()
    s_vres.reset()
    s_vth.reset()
    s_A.reset()
    s_vl.reset()
    plt.draw()
button.on_clicked(reset)

plt.show()

