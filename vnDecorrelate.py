import random
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np

def vn_convolve():
    '''
    performs a latency-free convolution
    '''
    return

def vn_generate(p, fs, dur):
    '''
    generates a velvet noise sequence
    p : density measured in spikes/impulses per second 
    fs : sample rate
    dur : duration of the sequence in seconds
    returns a list
    '''
    Ls = int(fs * dur) # size of the sequence in samples
    Td = fs / p # calculate the average spacing between two impulses (grid size)
    impulse_response = [0] * Ls
    for i, v in enumerate(impulse_response):
        v = 2 * round(random.random()) - 1 # sign of the impulse
        index = round(i * Td + random.random() * (Td - 1))
        if index < len(impulse_response):
            impulse_response[index] = v
    return impulse_response
#%%
DENSITY = 1500  # measured in spikes/impulses per second 
VNS_DURATION = 0.03 # duration of VNS in seconds
sample_rate, sig_int16 = wavfile.read("drums.wav")
vns = np.array(vn_generate(DENSITY, sample_rate, VNS_DURATION))
wavfile.write("velvet_noise.wav", sample_rate, vns)
plt.figure()
plt.plot(vns)
#plt.plot(sig_int16)