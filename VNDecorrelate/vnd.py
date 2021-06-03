import random as r
import numpy as np

# TODO:

# Finish documentation of VelvetNoise class
# Put example code in a seperate file, so vnd.py only requires numpy
# Implement Multiband decorrelation: higher frequencies -> more decorrelated

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

def rms_normalize(input_sig, output_sig):
    """
    Normalizes output_sig to the rms value of input_sig.

    Parameters
    ----------
    input_sig : numpy.ndarray
        The original dry signal.
    output_sig : numpy.ndarray
        The signal to normalize.

    Returns
    -------
    None.

    """
    output_sig *= np.sqrt(np.mean(np.square(input_sig))) \
        /  np.sqrt(np.mean(np.square(np.sum(output_sig, axis=1))))

def haas_delay(input_sig, delay_time, fs, channel, mode='LR'):
    """
    Return a stereo signal where the specified channel is delayed by delay_time.

    Parameters
    ----------
    input_sig : numpy.ndarray
        The input signal to apply the Haas Effect to.
    delay_time : float
        The time in ms to delay by.
    fs : int
        The sample rate.
    channel : int
        Which channel gets delayed. For stereo audio:
            0 is the left channel, 1 is the right channel.
            OR
            0 is the mid channel, 1 is the side channel.
    mode : string, optional
        Either 'LR' (Left-Right) or 'MS' (Mid-Side).
        If set to MS, then the delay will be applied to the mid or side channel
        depending on the channel argument.
        The default is 'LR'.

    Returns
    -------
    output_sig : numpy.ndarray
        The wet signal with one channel delayed.

    """

    # convert to 32 bit floating point, if it isn't already.
    if input_sig.dtype != np.float32:
        audio_sig = input_sig.astype(np.float32)
    else:
        audio_sig = input_sig
    # normalize so that the data ranges from -1 to 1 if it doesn't already.
    if np.max(np.abs(audio_sig)) > 1:
        audio_sig /= np.max(audio_sig)
    delay_len_samples = round(delay_time * fs)
    mono = False
    # if the input was mono, convert it to stereo.
    if input_sig.ndim != 2:
        mono = True
        audio_sig = np.column_stack((audio_sig, audio_sig))

    # if applicable, convert the left-right signal to a mid-side signal.
    # Mid-Side and L-R channel conversions:
    #    L = M + S
    #    R = M − S
    #    S = (L − R) / 2
    #    M = (L + R) / 2
    if mode == 'MS':
        mids = (audio_sig[:, 0] + audio_sig[:, 1]) * 0.5
        if mono: # sides will be silent
            sides = mids # this is techincally incorrect, but we fix it later.
        else:
            sides = (audio_sig[:, 0] - audio_sig[:, 1]) * 0.5
        # now stack them and store back into audio_sig
        audio_sig = np.column_stack((mids, sides))

    zero_padding_sig = np.zeros(delay_len_samples)
    wet_channel = np.concatenate((zero_padding_sig, audio_sig[:, channel]))
    dry_channel = np.concatenate((audio_sig[:, 0 if channel else 1], zero_padding_sig))

    # get the location of the wet and dry channels
    # then put them in a tuple so that they can be stacked
    location = (dry_channel, wet_channel) if channel else (wet_channel, dry_channel)
    audio_sig = np.column_stack(location)

    # convert back to left-right, if we delayed the mid and sides.
    if mode == 'MS':
        L = audio_sig[:, 0] + audio_sig[:, 1]
        R = audio_sig[:, 0] - audio_sig[:, 1]
        if mono:
            # we duplicated the mono channel, here we compensate for it.
            L *= 0.5
            R *= 0.5
        # now stack them and store back into audio_sig
        audio_sig = np.column_stack((L, R))

    return audio_sig

class VelvetNoise():
    def __init__(self, fs, p=1000, dur=0.03, lr=0.0197, ms=0.0096, seed=0):
        self.p = p  # density in impulses per second
        self.fs = fs  # sampling rate
        self.dur = dur  # duration in seconds
        self.num_outs = 2  # number of output channels
        self.vns = []  # velvet noise sequences
        self.impulses = []  # non-zero elements of the VNS
        self.lr = lr # left-right delay in ms
        self.ms = ms # mide-side delay in ms
        self.seed = seed  # the seed for the vns generator
        self.generate(self.p, self.dur)  # init filters

    def convolve(self, input_sig):
        """
        Perform the convolution of the velvet noise filters
        onto each channel of a signal.
        We take advantage of the sparse nature of the sequence
        to perform a latency-free convolution.

        Parameters
        ----------
        input_sig : numpy.ndarray
            The input signal to convolve with the generated filter.
        vns : numpy.ndarray
            A list containing the velvet noise filters
            to convolve with the input signal for each channel.

        Returns
        -------
        output_sig : numpy.ndarray
             the output signal in stereo
        """

        output_sig = np.zeros((len(input_sig), self.num_outs))
        for x, channel in enumerate(input_sig.T):
            matrix = np.zeros((len(self.impulses[x]), len(channel)))
            for m, k in enumerate(self.impulses[x].keys()):
                matrix[m, k:] += channel[:-k] if k else channel
            decay = list(self.impulses[x].values())
            output_sig[:, x] = np.sum(np.multiply(decay, matrix.T), axis=1)
        return output_sig / np.sum(np.abs(decay))

    def decorrelate(self, input_sig, num_outs, regenerate=False):
        self.num_outs = num_outs
        if regenerate:
            self.generate(self.p, self.dur)
        # convert to 32 bit floating point, if it isn't already
        # also normalize so that the data ranges from -1 to 1
        if input_sig.dtype != np.float32:
            input_sig = input_sig.astype(np.float32)
            input_sig /= np.max(np.abs(input_sig))
        # if the input is mono, then duplicate it to stereo before convolving
        if input_sig.ndim != self.num_outs:
            input_sig = np.column_stack((input_sig, input_sig))
        output_sig = input_sig + self.convolve(input_sig) * 0.5
        output_sig = haas_delay(output_sig, self.ms, self.fs, 1, mode='MS')
        output_sig = haas_delay(output_sig, self.lr, self.fs, 1, mode='LR')
        return output_sig

    def generate(self, p, dur):
        """
        Generate a velvet noise finite impulse response filter
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
        """
        if self.seed:
            r.seed(self.seed)
        self.vns = []
        self.impulses = []
        # size of the sequence in samples
        Ls = int(round(self.fs * dur))
        # average spacing between two impulses (grid size)
        Td = self.fs / p
        # total number of impulses
        M = int(Ls / Td)
        # coefficient values for segmented decay can be set manually
        si = [0.95, 0.5, 0.25, 0.1]  # decay coefficients
        I = len(si)  # number of segments
        # calculate the grid size between each impulse logarithmically
        # use a lambda function to apply this for each impulse index
        Tdl = lambda m: pow(10, m / M)
        # the calculated intervals between impulses
        intervals = np.array([Tdl(m) for m in range(M)])
        Tdl_max = np.sum(intervals)
        for _ in range(self.num_outs):
            vns = np.zeros((Ls,), np.float32)
            impulses = {}
            r1 = np.random.uniform(low=0, high=1, size=M)
            # first, randomize sign of the impulse
            s = (2 * np.round(r1)) - 1
            r2 = np.random.uniform(low=0, high=1, size=M)
            for m in range(M):
                # logarithmically distribute the impulses
                k = int(np.round(r2[m] * (Tdl(m) - 1)) \
                    + np.sum(intervals[:m]) * (Ls / Tdl_max))
                # store the index and corresponding scalar for segmented decay
                scalar = si[int(m / (M / I))] * s[m]
                if k < len(vns):
                    impulses[k], vns[k] = (scalar, scalar)
            self.vns.append(vns)
            self.impulses.append(impulses)

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wavfile

    VNS_DURATION = 0.03  # duration of VNS in seconds
    M = 120  # number of impulses
    DENSITY = int(M / VNS_DURATION)  # measured in impulses per second

    sample_rate, sig_float32 = wavfile.read("audio/guitar.wav")
    a = 2
    vnd = VelvetNoise(sample_rate, p=DENSITY, dur=VNS_DURATION, seed=a)
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
