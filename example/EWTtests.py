# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:00:03 2019

@author: John
"""
import numpy as np
import matplotlib.pyplot as plt
import ewtpy

signal = "sig3.csv" #sig1,sig2,sig3,eeg or any other csv file with a signal
data = np.loadtxt(signal, delimiter=",")
if len(data.shape) >1:
    f = data[:,0]
else:
    f = data
    
#f = f - np.mean(f)

N = 3 #number of supports
detect = "locmax" #detection mode: locmax, locmaxmin, locmaxminf
reg = 'none' #spectrum regularization - it is smoothed with an average (or gaussian) filter 
lengthFilter = 0 #length or average or gaussian filter
sigmaFilter = 0 #sigma of gaussian filter
Fs = 1 #sampling frequency, in Hz (if unknown, set 1)

ewt,  mfb ,boundaries = ewtpy.EWT1D(f, 
                                    N = N,
                                    log = 0,
                                    detect = detect, 
                                    completion = 0, 
                                    reg = reg, 
                                    lengthFilter = lengthFilter,
                                    sigmaFilter = sigmaFilter)

#plot original signal and decomposed modes
plt.figure()
plt.subplot(211)
plt.plot(f)
plt.title('Original signal %s'%signal)
plt.subplot(212)
plt.plot(ewt)
plt.title('EWT modes')

#%% show boundaries
ff = np.fft.fft(f)
freq=2*np.pi*np.arange(0,len(ff))/len(ff)

if Fs !=-1:
    freq=freq*Fs/(2*np.pi)
    boundariesPLT=boundaries*Fs/(2*np.pi)
else:
    boundariesPLT = boundaries

ff = abs(ff[:ff.size//2])#one-sided magnitude
freq = freq[:freq.size//2]

plt.figure()
plt.plot(freq,ff)
for bb in boundariesPLT:
    plt.plot([bb,bb],[0,max(ff)],'r--')
plt.title('Spectrum partitioning')
plt.xlabel('Hz')
plt.show()
