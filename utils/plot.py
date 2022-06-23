# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 18:42:27 2022

@author: Christian Konstantinov
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

from velvet_noise import VelvetNoise

# TODO: Plot Autocorrelogram and Cross Correlogram of Sine-sweep signal

def main():
    VNS_DURATION = 0.03  # duration of VNS in seconds
    M = 30  # number of impulses
    DENSITY = int(M / VNS_DURATION)  # measured in impulses per second

    sample_rate, sig_float32 = wavfile.read("audio/guitar.wav")
    vnd = VelvetNoise(sample_rate, p=DENSITY, dur=VNS_DURATION)
    result = vnd.decorrelate(sig_float32, num_outs=2)
    vns = np.array(vnd.vns).T

    wavfile.write("audio/guitar_dec.wav", sample_rate, result)

    plt.figure()
    plt.plot(vns[:, 0])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Velvet Noise Sequence L')

    plt.figure()
    plt.plot(vns[:, 1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Velvet Noise Sequence R')

    plt.figure()
    plt.plot(sig_float32)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Input')

    plt.figure()
    plt.plot(result)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Output')

if __name__ == '__main__':
    main()
