import random as r
import numpy as np

# TODO: 
# Finish decorrelate function 
    # Add LR LCR MS MS(LR) control options
# Add real-time support
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

def get_rms(input_sig, output_sig, axis=1):
    return np.sqrt(np.mean(np.square(input_sig))) \
        /  np.sqrt(np.mean(np.square(output_sig)))

class VelvetNoise():
    
    def __init__(self, fs, p=1000, dur=0.03):
        self.p = p # density in impulses per second
        self.fs = fs # sampling rate
        self.dur = dur # duration in seconds
        self.num_outs = 2 # number of output channels
        self.vns = [] # velvet noise sequence
        self.impulses = [] # non-zero elements of the VNS

    def convolve(self, input_sig):
        '''
        Performs the convolution of the velvet noise filters 
        onto each channel of a signal. 
        We take advantage of the sparse nature of the sequence
        to perform a latency-free convolution.
        
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
        output_sig = np.zeros((len(input_sig), self.num_outs))
        for x, channel in enumerate(input_sig.T):
            matrix = np.zeros((len(self.impulses[x]), len(channel)))
            for m, k in enumerate(self.impulses[x].keys()):
                if k != 0:
                    matrix[m,k:] += channel[:-k]
            decay = list(self.impulses[x].values())
            output_sig[:, x] = np.sum(np.multiply(decay, matrix.T), axis=1)
            output_sig[:, x] *= get_rms(input_sig[:, x], output_sig[:, x])
        return output_sig 
    @measure
    def decorrelate(self, input_sig, num_outs, a=0):
        self.num_outs = num_outs
        self.generate(self.p, self.dur, a=a)
        # if the input is mono, then duplicate it to stereo before convolving
        if input_sig.ndim != self.num_outs:
            input_sig = np.column_stack((input_sig, input_sig))
        # convert to 32 bit floating point, if it isn't already
        if input_sig.dtype != 'float32':
            input_sig = input_sig.astype(np.float32)    
        return self.convolve(input_sig)

    def generate(self, p, dur, a=0):
        '''
        Generates a velvet noise finite impulse response filter
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
        dur : float
            The duration of the sequence in seconds.
        a : int
            The seed for the random number generator.
        '''
        if a:
            r.seed(a)
        # size of the sequence in samples
        Ls = int(round(self.fs * dur))
        # average spacing between two impulses (grid size)
        Td = self.fs / p
        # total number of impulses
        M = int(Ls / Td)
        # coefficient values for segmented decay can be set manually
        si = [0.95, 0.5, 0.25, 0.1] # decay coefficients
        I = len(si) # number of segments
        # calculate the grid size between each impulse logarithmically
        # use a lambda function to apply this for each impulse index
        Tdl = lambda m : (Ls / 100) * pow(10, (m-2) / float(M))
        for _ in range(self.num_outs):
            vns = np.zeros((Ls,), np.float32)
            impulses = {}
            for m in range(M):
                # first, randomize sign of the impulse
                s = (2 * round(r.random()) - 1)
                # logarithmically distribute the impulses
                k = round(r.random() * (Tdl(m) - 1)) \
                    + int(np.floor(sum((Tdl(i) for i in range(m)))))
                # store the index and corresponding scalar for segmented decay
                scalar = si[int(m / (M / I))] * s
                if k < len(vns):
                    impulses[k], vns[k] = (scalar, scalar)
            self.vns.append(vns)
            self.impulses.append(impulses)
    
    def np_convolve(self, input_sig, channel):
        result = np.convolve(input_sig, self.vns[channel], mode='same')
        result *= get_rms(input_sig, result)
        return result
#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wavfile
    import scipy.signal as s
    
    VNS_DURATION = 0.015 # duration of VNS in seconds
    M = 30 # number of impulses
    DENSITY = int(M / VNS_DURATION) # measured in impulses per second 
    
    sample_rate, sig_float32 = wavfile.read("audio/vocal.wav")
    vnd = VelvetNoise(sample_rate, p=DENSITY)
    #a = 2
    a = 0
    result = vnd.decorrelate(sig_float32, num_outs=2, a=a)
    vns = np.array(vnd.vns).T
    
    @measure
    def np_decorrelate():
        return np.array([vnd.np_convolve(sig_float32, 0), vnd.np_convolve(sig_float32, 1)]).T
    #result = np_decorrelate()
    
    @measure
    def scipy_decorrelate():
        return s.oaconvolve(sig_float32, vns)
    #sig_float32 = np.column_stack((sig_float32, sig_float32))
    #result = scipy_decorrelate()
    
    # Mid-Side and L-R channel conversions
    # M = (L + R) / 2 , S = (L - R) / 2
    # L = M + S , R = M - S
    Sides = result[:, 1]
    Mids = sig_float32
    L = result[:, 0]
    R = result[:, 1]
    
    # Mid-Side Decorrelation
    result = np.column_stack([Mids + Sides, Mids - Sides])
    result *= get_rms(sig_float32, result) 
    wavfile.write("audio/vocal_decorrelated1.wav", sample_rate, result)
    
    # L-C-R Decorrelation
    result = np.column_stack([L + Mids, R + Mids])
    result *= get_rms(sig_float32, result) 
    wavfile.write("audio/vocal_decorrelated2.wav", sample_rate, result)
    
    # Mid-Side using L and R channels
    Sides = (L - R) / 2
    result = np.column_stack([Mids + Sides, Mids - Sides])
    result *= get_rms(sig_float32, result) 
    
    wavfile.write("audio/vocal_decorrelated3.wav", sample_rate, result) 

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