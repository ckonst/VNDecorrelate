import numpy as np

from timing import timed

from typing import List, Dict

# TODO:
# Implement Multiband decorrelation: higher frequencies -> more decorrelated
# Abstraction of the Filter and Decorrelator (?)
# Decouple Haas Delay and Velvet Noise

def rms_normalize(input_sig, output_sig):
    """
    Normalizes output_sig in-place to the rms value of input_sig.

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
    """
    A velvet noise decorrelator for audio. See method docstrings for more details.

    Attributes:
        fs : int
            The sample rate.
        p : int
            Impulse density in impulses per second.
        dur : float
            The duration of the velvet noise sequence in seconds.
        width : float
            Controls the level of the side channel, as a percent of both the Mid-Side channels.
        num_outs : int
            The number of output channels.
        vns : List[numpy.ndarray]
            Velvet noise sequences, one for each channel.
            Mainly for plotting with matplotlib or naive convolution with numpy.convolve
        impulses : List[Dict[int, int]]
            For each channel, store locations of nonzero impulses mapped to their sign (1 or -1).
        lr : float
            Left-Right delay in ms.
        ms : float
            Mid-side delay in ms.

    """

    def __init__(self, fs, p=1000, dur=0.03, lr=0.0197, ms=0.0096, width=0.5):
        """
        Impulse density, p, and filter length, dur, are chosen from the velvet noise paper.
        Left-Right and Mid-Side delay values can be chosen to reduce phase incoherence.
        In general, keeping these values under 20ms helps reduce audible doubling artifacts, which are more noticible in mono.

        """

        self.fs = fs
        self.p = p
        self.dur = dur
        self.width = width
        self.num_outs = 2
        self.vns: List[np.ndarray] = []
        self.impulses: List[Dict[int, int]] = []
        self.lr = lr
        self.ms = ms
        self.generate(self.p, self.dur)

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

        Returns
        -------
        output_sig : numpy.ndarray
             the output signal in stereo

        """

        sig_len = len(input_sig)
        output_sig = np.zeros((sig_len, self.num_outs))
        for x, channel in enumerate(input_sig.T):
            matrix = np.zeros((len(self.impulses[x]), sig_len))
            for m, k in enumerate(self.impulses[x].keys()):
                matrix[m, :-k if k else sig_len] += channel[k:]
            decay = list(self.impulses[x].values())
            output_sig[:, x] = np.sum(decay * matrix.T, axis=1)
        rms_normalize(input_sig, output_sig)
        return output_sig

    def decorrelate(self, input_sig, num_outs, regenerate=False, segmented_decay=True, log_distribution=True):
        """
        Perform a velvet noise decorrelation on input_sig with num_outs channels.
        As of right now only supports stereo decorrelation (i.e. no quad-channel, 5.1, 7.1, etc. support)
        This method will perform an optimized velvet noise convolution for each channel
        to generate the side channel content.
        Then it will interpolate between the mid (dry) and side (wet) channels at self.width.
        Lastly, this method will apply a delay to the mid and left channels, and return the result.

        Parameters
        ----------
        input_sig : numpy.ndarray
            The mono or stereo input signal to upmix or decorrelate.
        num_outs : int
            The number of output channels.
        regenerate : bool, optional
            Whether or not to call self.generate to replace the velvet noise filters. The default is False.
        segmented_decay : bool, optional
            Whether or not self.generate (if used) uses a segmented decay envelope. The default is True.
        log_distribution : bool, optional
            Whether or not self.generate (if used) uses logarithmic impulse distribution. The default is True.

        Returns
        -------
        output_sig : numpy.ndarray
            The stereoized, decorrelated output signal.

        """

        self.num_outs = num_outs
        if regenerate:
            self.generate(self.p, self.dur, segmented_decay, log_distribution)
        # convert to 32 bit floating point, if it isn't already
        # also normalize so that the data ranges from -1 to 1
        if input_sig.dtype != np.float32:
            input_sig = input_sig.astype(np.float32)
            input_sig /= np.max(np.abs(input_sig))
        # if the input is mono, then duplicate it to stereo before convolving
        if input_sig.ndim != self.num_outs:
            input_sig = np.column_stack((input_sig, input_sig))
        output_sig = self.convolve(input_sig)
        output_sig[:, 0] = -output_sig[:, 0] * self.width
        output_sig += input_sig * (1 - self.width)
        output_sig = haas_delay(output_sig, self.ms, self.fs, 1, mode='MS')
        output_sig = haas_delay(output_sig, self.lr, self.fs, 1, mode='LR')
        return output_sig

    @timed
    def generate(self, p, dur, segmented_decay=True, log_distribution=True):
        """
        Generate a velvet noise finite impulse response filter
        to convolve with an input signal. Overwrites self.vns and self.impulses.
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
        segmented_decay : bool, optional
            Whether or not to use a segmented decay envelope. The default is True.
        log_distribution : bool, optional
            Whether or not to use logarithmic impulse distribution. The default is True.

        """

        self.vns = []
        self.impulses = []
        # size of the sequence in samples
        Ls = int(round(self.fs * dur))
        # average spacing between two impulses (grid size)
        Td = self.fs / p
        # total number of impulses
        M = int(Ls / Td)
        # coefficient values for segmented decay can be set manually
        si = [0.95, 0.6, 0.25, 0.1]  # decay coefficients
        I = len(si)  # number of segments
        # calculate the grid size between each impulse logarithmically
        # use a lambda function to apply this for each impulse index
        Tdl = lambda m: pow(10, m / M)
        # impulse location function without logorithmic impulse distribution
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
            k = 0
            for m in range(M):
                if log_distribution:
                    # logarithmically distribute the impulses
                    k = int(np.round(r2[m] * (Tdl(m) - 1)) \
                        + np.sum(intervals[:m]) * (Ls / Tdl_max))
                else:
                    k = int(np.round(m * Td + r2[m] * (Td - 1)))
                if segmented_decay:
                    # store the index and corresponding scalar for segmented decay
                    scalar = si[int(m / (M / I))] * s[m]
                else: scalar = s[m]
                if k < len(vns):
                    impulses[k], vns[k] = (scalar, scalar)
            self.vns.append(vns)
            self.impulses.append(impulses)

if __name__ == '__main__':
    from plot import main; main()
