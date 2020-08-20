import random as r
import numpy as np

# TODO: 
# Finish decorrelate function 
    # Test L-R decorrelation vs. Mid-Side decorrelation
    # Test for stereo to stereo decorrelation
# Finish documentation of VelvetNoise class
# Put example code in a seperate file, so vnd.py only requires numpy
# Try low pass filtering input signal at 3.5kHz

from functools import wraps
from time import time

def measure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {} ms'.format((end-start) * 1000))
        return result
    return wrapper

def tofloat32(input_sig):
    '''
    Converts an input signal to a 32 bit floating point format

    Parameters
    ----------
    input_sig : n-dimension numpy array
        The input to convert to a 32 bit floating point array.

    Returns
    -------
    input_sig : n-dimension numpy array
        The signal converted to 32 bit floating point array.
    '''
    input_sig = input_sig.astype(np.float32, copy=False)
    input_sig = normalize(input_sig)
    return input_sig

def restore(channel, original):
    '''
    Restores a channel of audio to the original level of the input
    after a convolution.

    Parameters
    ----------
    channel : numpy array
        The channel to restore.
    original : numpy array
        The original input to match the level to.

    Returns
    -------
    numpy array
        the channel normalized to the level of the original.

    '''
    return normalize(channel, min(original), max(original))

def normalize(channel, min=-1, max=1):
    '''
    Takes in an input channel and normalizes it to the min and max values.
    
    Parameters
    ----------
    channel : numpy array
        A single channel of the signal to normalize.
    min : int, optional
        The minimum value to set the channel to. The default is -1.
    max : int, optional
        The maximum value to set the channel to. The default is 1.

    Returns
    -------
    channel : numpy array
        The normalized channel.

    '''
    #normalize to -1 to 1 first
    if np.max(np.abs(channel)) != 0:
        channel /= np.max(np.abs(channel))
    if max != 1 and min != -1:
        pos = channel.clip(min=0) * max
        neg = channel.clip(max=0) * -min
        channel = pos + neg
    return channel

class VelvetNoise():
    def __init__(self, p=1000, fs=44100, dur=0.03):
        self.p = p
        self.fs = fs
        self.dur = dur
        self.vns = []
        self.impulses = []
        # coefficient values for segmented decay can be set manually
        self.si = [0.95, 0.5, 0.25, 0.1]
        self.I = len(self.si)
        
    @measure
    def convolve(self, input_sig, num_channels=2):
        '''
        Performs the convolution of the velvet noise filters 
        onto each channel of a signal. 
        We take advantage of the sparse nature of the sequence
        to do the convolution very quickly.
        
        Parameters
        ----------
        input_sig : 2d numpy array
            The input signal to convolve with the generated filter.
        vns : 2d numpy array
            A list containing the velvet noise filters 
            to convolve with the input signal for each channel.
        
        Returns
        ----------
        output_sig : 2d numpy array
             the output signal in stereo
        '''
        output_sig = np.zeros((len(input_sig), num_channels))
        for x, channel in enumerate(input_sig.T):
            matrix = np.zeros((len(self.impulses[x]), len(channel)))
            for m, k in enumerate(self.impulses[x].keys()):
                if k != 0:
                    matrix[m,k:] += channel[:-k]
            output_sig[:,x] = np.sum(np.multiply(list(self.impulses[x].values()), matrix.T), axis=1)
            output_sig[:,x] = restore(output_sig[:,x], input_sig[:,x])
        return output_sig 
    
    def decorrelate(self, input_sig, num_channels=2):
        self.vns = []
        self.impulses = []
        for i in range(num_channels):
            self.generate(self.p, self.fs, self.dur)
        # if the input is mono, then duplicate it to stereo before convolving
        if input_sig.ndim == 1:
            input_sig = np.column_stack((input_sig, input_sig))
        # convert to 32 bit floating point, if it isn't already
        if input_sig.dtype != 'float32':
            input_sig = tofloat32(input_sig)        
        return self.convolve(input_sig, num_channels)
    @measure
    def generate(self, p, fs, dur):
        '''
        Generates a velvet noise sequence as a finite impulse response filter
        to convolve with an input signal.
        To avoid audible smearing of transients the following are applied:
            - A segmented decay envelope.
            - Logarithmic impulse distribution.
        Segmented decay is preferred to exponential decay because it uses
        less multiplication, and produces satisfactory results.
    
        Parameters
        ----------
        p : int
            The density measured in spikes/impulses per second.
        fs : int
            The sample rate.
        dur : float
            The duration of the sequence in seconds.
        
        Returns
        ----------
        vns: numpy array
            The generated velvet noise sequence.
        
        '''
        # size of the sequence in samples
        Ls = int(round(fs * dur))
        # average spacing between two impulses (grid size)
        Td = fs / p
        # total number of impulses
        M = int(Ls / Td)
        # calculate the grid size between each impulse logarithmically
        # use a lambda function to be able to apply this for each impulse index
        Tdl = lambda m : (Ls / 100) * pow(10, (m-2) / float(M))
        vns = np.zeros((Ls,), np.float32)
        impulses = {}
        for m in range(M):
            # first, randomize sign of the impulse
            s = (2 * round(r.random()) - 1)
            # logarithmically distribute the impulses
            k = round(r.random() * (Tdl(m) - 1)) \
                + int(np.floor(sum((Tdl(i) for i in range(m)))))
            # store the index and corresponding scalar for segmented decay
            scalar = self.si[int(m / (M / self.I))] * s
            if k < len(vns):
                impulses[k], vns[k] = (scalar, scalar)
        print(len(impulses))
        self.vns.append(vns)
        self.impulses.append(impulses)
   
    def conv(self, input_sig, channel):
        result = np.convolve(input_sig, self.vns[channel], mode='same')
        return restore(result, input_sig)
#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wavfile
    
    # Make an adjustable parameter
    VNS_DURATION = 0.03 # duration of VNS in seconds
    
    # Make internal to the class
    M = 30 # number of impulses
    DENSITY = int(M / VNS_DURATION) # measured in impulses per second 
    
    sample_rate, sig_float32 = wavfile.read("audio/vocal.wav")
    #r.seed(a=6)
    vnd = VelvetNoise(DENSITY, sample_rate, VNS_DURATION)
    result = vnd.decorrelate(sig_float32)
    
    @measure
    def np_convolve():
        vnd.generate(DENSITY, sample_rate, VNS_DURATION)
        vnd.generate(DENSITY, sample_rate, VNS_DURATION)
        return np.array([vnd.conv(sig_float32, 0), vnd.conv(sig_float32, 1)]).T
    #result = np_convolve()
    
    # By doing this convolution we calculate the side content,
    # given a mono signal.
    # convert Mid-Side channels to L-R channels 
    # L = M + S , R = M - S
    #result = np.array([np.add(sig_float32, result), np.subtract(sig_float32, result)])
    #result = np.reshape(result, (int(result.size/2), 2))
    vns = vnd.vns

    wavfile.write("audio/vocal_decorrelated.wav", sample_rate, result) 
    fs, sf32 = wavfile.read("audio/correct.wav")

    plt.figure()
    plt.plot(vns[0])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Velvet Noise Sequence L')
    
    plt.figure()
    plt.plot(vns[1])
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
    
    plt.figure()
    plt.plot(sf32)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Correct Solution')