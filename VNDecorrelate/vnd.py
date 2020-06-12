import random
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np
from functools import wraps
from time import time

#TODO: convolve each channel with a unique VNS

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it

@measure
def vn_convolve(input_sig, impulse_response):
    '''
    performs a segmented convolution
    of the velvet noise sequence with a signal
    we take advantage of the sparse nature of the sequence
    to do convolution very quickly
    
    Parameters
    ----------
    input_sig : numpy array
        the input signal to convolve with the VNS
    impulse_response : 2d numpy array
        a list containing the velvet noise sequences 
        to convolve with the input signal for each cahnnel
    
    Returns
    ----------
    output_sig : numpy array
         the output signal in stereo
    '''
    #if the input is mono, then duplicate it to stereo before convolving
    if input_sig.ndim == 1:
        input_sig = np.column_stack((input_sig, input_sig))
    output_sig = np.empty((len(input_sig), 2)) 
    # perform the convolution
    # for each channel
    for i, channel in enumerate(input_sig.T):
        # for further optimization, store the non-zero values
        # of the VNS in two seperate arrays
        k_pos = []
        k_neg = []  
        if i < len(impulse_response):
            for j, v in enumerate(impulse_response[i]):
                if v == 0:
                    continue
                elif v > 0:
                    k_pos.append(i)
                else:
                    k_neg.append(i)
        # for each sample in the input signal
        for n, sample in enumerate(input_sig):
            pos_sum = sum(channel[n - imp] for m, imp in enumerate(k_pos))
            neg_sum = sum(channel[n - imp] for m, imp in enumerate(k_neg))
            output_sig[n][i] = (pos_sum - neg_sum) * 0.27
    return output_sig

@measure
def vn_generate(p, fs, dur):
    '''
    generates a velvet noise sequence
    with segmented decay so that the transients
    are not smeared
    
    Parameters
    ----------
    p : int
        density measured in spikes/impulses per second 
    fs : int
        sample rate
    dur : float
        duration of the sequence in seconds
    
    Returns
    ----------
    impulse_response: numpy array
        the generated velvet noise sequence
    
    '''
    # size of the sequence in samples
    Ls = int(fs * dur) 
    # calculate the average spacing between two impulses (grid size)
    Td = fs / p
    # number of segments
    I = 4
    # coefficient values to applay segmented decay
    si = [0.95, 0.5, 0.25, 0.1]
    impulse_response = np.zeros((Ls,))
    for j, v in enumerate(impulse_response):
        # first, randomize sign of the impulse
        v = (2 * round(random.random()) - 1)
        # now randomize the distance between impulses
        index = round(j * Td + random.random() * (Td - 1))
        if index < len(impulse_response):
            impulse_response[index] = v * si[int(index / (Ls / I))]
        
    return impulse_response

#%%
DENSITY = 1000  # measured in spikes/impulses per second 
VNS_DURATION = 0.03 # duration of VNS in seconds
LR_DELAY = 0.02
sample_rate, sig_float32 = wavfile.read("drums.wav")
vns = [vn_generate(DENSITY, sample_rate, VNS_DURATION), vn_generate(DENSITY, sample_rate, VNS_DURATION)]
result = vn_convolve(sig_float32, vns)
wavfile.write("drums_decorrelated.wav", sample_rate, result)

plt.figure()
plt.plot(vns[0])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('FIR')

plt.figure()
plt.plot(vns[1])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('FIR')

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