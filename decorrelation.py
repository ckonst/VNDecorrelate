# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:18:33 2022

@author: Christian Konstantinov
"""

import numpy as np

from abc import ABC, abstractmethod
from typing import Mapping, Sequence
from utils import dsp
from utils.timing import timed

# ----------------------------------------------------------------------------
#
# Abstract Decorrelator Class
#
# ----------------------------------------------------------------------------

class Decorrelator(ABC):

    def __init__(self, fs: int = 44100, num_ins: int = 2, num_outs: int = 2, width: float = 0.5):
        self.fs = fs
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.width = width

    @abstractmethod
    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, input_sig: np.ndarray) -> np.ndarray:
        return self.decorrelate(input_sig)

# ----------------------------------------------------------------------------
#
# Signal Chain Class
#
# ----------------------------------------------------------------------------

class SignalChain(Decorrelator):

    def __init__(self, decorrelators: Sequence[Decorrelator], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decorrelators = decorrelators

    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        for decorrelator in self.decorrelators:
            (cascaded_sig := decorrelator(input_sig))
        return cascaded_sig

# ----------------------------------------------------------------------------
#
# Haas Effect Decorrelator
#
# ----------------------------------------------------------------------------

class HaasEffect(Decorrelator):
    def __init__(self, delay_time=0.02, channel=0, mode='LR', *args, **kwargs):
        """Left-Right and Mid-Side delay values can be chosen to reduce phase incoherence.

        In general, keeping these values under 20ms helps reduce audible doubling artifacts, which are more noticible in mono."""
        super().__init__(*args, **kwargs)
        self.delay_time = delay_time
        self.channel = channel
        self.mode = mode

    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        return haas_delay(input_sig, self.delay_time, self.fs, self.channel, mode=self.mode)

# TODO: consolidate with HaasEffect class
def haas_delay(input_sig, delay_time, fs, channel, mode='LR'):
    """Return a stereo signal where the specified channel is delayed by delay_time.

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
    audio_sig = dsp.to_float32(input_sig)
    # normalize so that the data ranges from -1 to 1 if it doesn't already.
    dsp.peak_normalize(audio_sig)

    delay_len_samples = round(delay_time * fs)
    mono = False

    # if the input was mono, convert it to stereo.
    if input_sig.ndim != 2:
        mono = True
        audio_sig = dsp.mono_to_stereo(audio_sig)

    # if applicable, convert the left-right signal to a mid-side signal.
    if mode == 'MS':
        if mono: # sides will be silent
            mids = dsp.stereo_to_mono(audio_sig)
            sides = mids # this is techincally incorrect, but we fix it later.
            # now stack them and store back into audio_sig
            audio_sig = np.column_stack((mids, sides))
        else:
            audio_sig = dsp.LR_to_MS(audio_sig)

    zero_padding_sig = np.zeros(delay_len_samples)
    wet_channel = np.concatenate((zero_padding_sig, audio_sig[:, channel]))
    dry_channel = np.concatenate((audio_sig[:, -(channel - 1)], zero_padding_sig))

    # get the location of the wet and dry channels
    # then put them in a tuple so that they can be stacked
    location = (dry_channel, wet_channel) if channel else (wet_channel, dry_channel)
    audio_sig = np.column_stack(location)

    # convert back to left-right, if we delayed the mid and sides.
    if mode == 'MS':
        audio_sig = dsp.MS_to_LR(audio_sig)
        if mono:
            # we duplicated the mono channel, here we compensate for it.
            audio_sig *= 0.5

    return audio_sig

# ----------------------------------------------------------------------------
#
# Velvet Noise Decorrelator
#
# ----------------------------------------------------------------------------

# TODO:
# Optimize Velvet Noise convolution
# Give variables better names
# Implement Multiband decorrelation: higher frequencies -> more decorrelated

class VelvetNoise(Decorrelator):
    """A velvet noise decorrelator for audio.

    See method docstrings for more details.

    Attributes:
        density : int
            Impulse density in impulses per second.
        length : float
            The duration of the velvet noise sequence in seconds.
        width : float
            Controls the level of the side channel, as a percent of both the Mid-Side channels.
        num_outs : int
            The number of output channels.
        vns : List[numpy.ndarray]
            Velvet noise sequences, one for each channel.
            Mainly for plotting with matplotlib or naive convolution with numpy.convolve
        impulses : Sequence[Mapping[int, int]]
            For each channel, store locations of nonzero impulses mapped to their sign (1 or -1).

    """

    def __init__(self, density=1000, length=0.03, *args, **kwargs):
        """Impulse density, and filter length defaults are chosen from the velvet noise paper.

        """
        super().__init__(*args, **kwargs)
        self.density = density
        self.length = length
        self.vns: Sequence[np.ndarray] = []
        self.impulses: Sequence[Mapping[int, int]] = []
        self.generate(self.p, self.dur)

    def convolve(self, input_sig):
        """Perform the convolution of the velvet noise filters onto each channel of a signal.

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
        dsp.rms_normalize(input_sig, output_sig)
        return output_sig

    def decorrelate(self, input_sig, segmented_decay=True, log_distribution=True):
        """Perform a velvet noise decorrelation on input_sig with num_outs channels.

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
        segmented_decay : bool, optional
            Whether or not self.generate (if used) uses a segmented decay envelope. The default is True.
        log_distribution : bool, optional
            Whether or not self.generate (if used) uses logarithmic impulse distribution. The default is True.

        Returns
        -------
        output_sig : numpy.ndarray
            The stereoized, decorrelated output signal.

        """

        # convert to 32 bit floating point, if it isn't already
        input_sig = dsp.to_float32(input_sig)
        # normalize so that the data ranges from -1 to 1
        dsp.peak_normalize(input_sig)
        # if the input is mono, then duplicate it to stereo before convolving
        if input_sig.ndim == 1:
            input_sig = dsp.mono_to_stereo(input_sig)
        output_sig = self.convolve(input_sig)
        output_sig[:, 0] = -output_sig[:, 0] * self.width
        output_sig += input_sig * (1 - self.width)
        return output_sig

    # TODO: Remove side effects
    # TODO: return positive and negative impulse sequences
    def generate(self, segmented_decay=True, log_distribution=True):
        """Generate a velvet noise finite impulse response filter to convolve with an input signal.

        Overwrites self.vns and self.impulses.
        To avoid audible smearing of transients the following are applied:
            - A segmented decay envelope.
            - Logarithmic impulse distribution.
        Segmented decay is preferred to exponential decay because it uses
        less multiplication, and produces satisfactory results.

        Parameters
        ----------
        segmented_decay : bool, optional
            Whether or not to use a segmented decay envelope. The default is True.
        log_distribution : bool, optional
            Whether or not to use logarithmic impulse distribution. The default is True.

        """

        self.vns = []
        self.impulses = []
        sequence_len = int(round(self.fs * self.length))
        # average spacing between two impulses (grid size)
        grid_size = self.fs / self.density
        num_impulses = int(sequence_len / grid_size)
        # coefficient values for segmented decay can be set manually
        decay_coeffs = [0.85, 0.55, 0.35, 0.2]
        num_segments = len(decay_coeffs)
        # calculate the grid size between each impulse logarithmically
        # use a function to apply this for each impulse index
        def log_grid_size(m): return pow(10, m / num_impulses)
        # the calculated intervals between impulses
        intervals = np.array([log_grid_size(m) for m in range(num_impulses)])
        sum_intervals = np.sum(intervals)
        for _ in range(self.num_outs):
            vns = np.zeros((sequence_len,), np.float32)
            impulses = {}
            r1 = np.random.uniform(low=0, high=1, size=num_impulses)
            # first, randomize sign of the impulse
            sign = (2 * np.round(r1)) - 1
            r2 = np.random.uniform(low=0, high=1, size=num_impulses)
            k = 0
            for m in range(num_impulses):
                if log_distribution:
                    k = int(np.round(r2[m] * (log_grid_size(m) - 1)) \
                        + np.sum(intervals[:m]) * (sequence_len / sum_intervals))
                else:
                    # impulse location function without logorithmic impulse distribution
                    k = int(np.round(m * grid_size + r2[m] * (grid_size - 1)))
                if segmented_decay:
                    # store the index and corresponding scalar for segmented decay
                    scalar = decay_coeffs[int(m / (num_impulses / num_segments))] * sign[m]
                else: scalar = sign[m]
                impulses[k], vns[k] = scalar, scalar
            self.vns.append(vns)
            self.impulses.append(impulses)

@timed
def main():
    import scipy.io.wavfile as wavfile
    fs, sig_float32 = wavfile.read("audio/vocal.wav")
    decorrelators = (
        VelvetNoise(p=1000, fs=fs, dur=0.03, width=0.5),
        HaasEffect(0.0197, fs=fs, channel=1, mode='LR'),
        HaasEffect(0.0096, fs=fs, channel=1, mode='MS'))
    chain = SignalChain(decorrelators, fs, 1, 2)
    chain(sig_float32)

if __name__ == '__main__':
    main()
