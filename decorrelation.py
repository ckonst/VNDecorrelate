# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:18:33 2022

@author: Christian Konstantinov
"""

import numpy as np

from abc import ABC, abstractmethod
from typing import List, Sequence
from utils import dsp

# ----------------------------------------------------------------------------
#
# Abstract Decorrelator Class
#
# ----------------------------------------------------------------------------

class Decorrelator(ABC):

    """An abstract base class for a Decorrelator.

    Attributes:
        fs : int
            The sampling frequency of the signal. Default is 44100.
        num_ins : int
            The number of input channels. Default is 2.
        num_outs : int
            The number of output channels. Default is 2.
        width : float
            Controls the level of the side channel, as a percent of both the Mid-Side channels [0.0, 1.0]. Default is 1.0.

    """

    def __init__(self, fs: int = 44100, num_ins: int = 2, num_outs: int = 2, width: float = 0.5):
        """Initialize all common parameters."""
        self.fs = fs
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.width = width

    @abstractmethod
    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        """Main decorrelation function that must be implemented in subclasses."""
        pass

    def __call__(self, input_sig: np.ndarray) -> np.ndarray:
        """Alternative way of calling decorrelate."""
        return self.decorrelate(input_sig)

# ----------------------------------------------------------------------------
#
# Signal Chain Class
#
# ----------------------------------------------------------------------------

# TODO: deferred instantiation?
class SignalChain(Decorrelator):

    """A simple class for cascading decorrelators.

    Attributes:
        decorrelators : Sequence[Decorrelator]
            The sequence containing the Decorrelator objects whose decorrelate methods will be called.

    """

    def __init__(self, decorrelators: Sequence[Decorrelator], *args, **kwargs):
        """Initialize the same way any other decorrelator would."""
        super().__init__(*args, **kwargs)
        self.decorrelators = decorrelators

    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        """Decorrelate input_sig with all decorrelators in this SignalChain.

        Parameters
        ----------
        input_sig : np.ndarray
            The original signal.

        Returns
        -------
        cascaded_sig : np.ndarray
            The signal processed by cascading decorrelators.

        """
        for decorrelator in self.decorrelators:
            (cascaded_sig := decorrelator(input_sig))
        return cascaded_sig

# ----------------------------------------------------------------------------
#
# Haas Effect Decorrelator
#
# ----------------------------------------------------------------------------

class HaasEffect(Decorrelator):

    """A Decorrelator that utilizes the Haas Effect.

    In music, the Haas Effect delays a channel or channels by a small amount to alter the cross correlation between each channel.
    For stereo audio, this can be easily done on either the left or right channel, or on the mid or side channel for different results.

    Attributes:
        delay_time : float
            The time in seconds to delay the channel by.
        channel : int
            Which channel gets delayed. For stereo audio:
                0 is the left channel, 1 is the right channel.
                OR
                0 is the mid channel, 1 is the side channel.
        mode : str
            Either 'LR' (Left-Right) or 'MS' (Mid-Side).
            If set to MS, then the delay will be applied to the mid or side channel
            depending on the channel argument.
            The default is 'LR'.

    """

    def __init__(self, delay_time: float = 0.02, channel: int = 0, mode: str = 'LR', width: float = 0.5, *args, **kwargs):
        """Left-Right and Mid-Side delay values can be chosen to reduce phase incoherence.

        In general, keeping these values under 20ms helps reduce audible doubling artifacts,
        which are more noticible in mono.

        """
        super().__init__(width=width, *args, **kwargs)
        self.delay_time = delay_time
        self.channel = channel
        self.mode = mode

    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        """Perform a Haas Effect decorrelation on input_sig.

        Parameters
        ----------
        input_sig : np.ndarray
            The mono or stereo input signal to upmix or decorrelate.

        Returns
        -------
        output_sig : np.ndarray
            The stereoized, decorrelated output signal.

        """
        # convert to 32 bit floating point, if it isn't already
        input_sig = dsp.to_float32(input_sig)
        # normalize so that the data ranges from -1 to 1 if it doesn't already
        dsp.peak_normalize(input_sig)
        # perform the decorrelation
        output_sig = self.haas_delay(input_sig)
        # adjust the width parameter.
        output_sig = dsp.apply_stereo_width(output_sig)
        return output_sig

    def haas_delay(self, input_sig: np.ndarray) -> np.ndarray:
        """Return a stereo signal where the specified channel is delayed by delay_time.

        Parameters
        ----------
        input_sig : numpy.ndarray
            The input signal to apply the Haas Effect to.
        delay_time : float
            The time in seconds to delay by.
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
        output_sig = input_sig
        delay_len_samples = round(self.delay_time * self.fs)
        mono = False

        # if the input was mono, convert it to stereo.
        if input_sig.ndim == 1:
            mono = True
            output_sig = dsp.mono_to_stereo(input_sig)

        # if applicable, convert the left-right signal to a mid-side signal.
        if self.mode == 'MS':
            if mono: # sides will be silent
                mids = dsp.stereo_to_mono(output_sig)
                sides = mids # this is techincally incorrect, but we fix it later.
                # now stack them and store back into audio_sig
                output_sig = np.column_stack((mids, sides))
            else:
                output_sig = dsp.LR_to_MS(output_sig)

        zero_padding_sig = np.zeros(delay_len_samples)
        wet_channel = np.concatenate((zero_padding_sig, output_sig[:, self.channel]))
        dry_channel = np.concatenate((output_sig[:, -(self.channel - 1)], zero_padding_sig))

        # get the location of the wet and dry channels
        # then put them in a tuple so that they can be stacked
        location = (dry_channel, wet_channel) if self.channel else (wet_channel, dry_channel)
        output_sig = np.column_stack(location)

        # convert back to left-right, if we delayed the mid and sides.
        if self.mode == 'MS':
            output_sig = dsp.MS_to_LR(output_sig)
            if mono:
                # we duplicated the mono channel, here we compensate for it.
                output_sig *= 0.5

        return output_sig

# ----------------------------------------------------------------------------
#
# Velvet Noise Decorrelator
#
# ----------------------------------------------------------------------------

# TODO:
# Implement Multiband decorrelation: higher frequencies -> more decorrelated
class VelvetNoise(Decorrelator):

    """A velvet noise decorrelator for audio.

    See method docstrings for more details.

    Attributes:
        density : int
            Impulse density in impulses per second. Default is 1000.
        duration : float
            The duration of the velvet noise sequence in seconds. Default is 0.03.
        segment_scalars : Sequence[float]
            The sequence of coefficients for segmented decay, one for each segment.
        seed : int
            The seed for the velvet noise generator.

    """

    def __init__(self, duration: float = 0.03, num_impulses: int = 30, segment_scalars: Sequence[float] = None,
                 log_distribution=True, seed: int = None, *args, **kwargs):
        """Impulse density, and filter length defaults are chosen from the velvet noise paper."""
        super().__init__(*args, **kwargs)
        self.num_impulses = num_impulses
        self.duration = duration
        if segment_scalars is None:
            self.segment_scalars = (0.85, 0.55, 0.35, 0.2)
        else:
            self.segment_scalars = segment_scalars
        self.log_distribution = log_distribution
        self.seed = seed
        '''
        _vn_sequences is of shape (num_outs, num_segments, 2, num_positive/num_negative)

        It maps each channel to a list of equal length segments, determined by segment_scalars.
        Each segment is then split into lists of negative and positive impulses at indices 0 and 1 respectively.
        Because of the inhomogeneity of the last dimension, this cannot be converted to a numpy array.
        Dimension 2 (of size 2) could be of Tuples, but for simplicity,
        I use Lists since the rest of the sequence must be generated with Lists.
        '''
        self._vn_sequences: List[List[List[List[int]]]] = self._generate()

    def convolve(self, input_sig: np.ndarray) -> np.ndarray:
        """Perform the convolution of the velvet noise filters onto each channel of a signal.

        We take advantage of the sparse nature of the sequence
        to perform a latency-free convolution.

        Parameters
        ----------
        input_sig : np.ndarray
            The input signal to convolve with the generated filter.

        Returns
        -------
        output_sig : np.ndarray
             The stereoized, decorrelated output signal.

        """
        sig_len = len(input_sig)
        output_sig = np.zeros((sig_len, self.num_outs))
        for ci, channel in enumerate(input_sig.T):
            for si, segment in enumerate(self._vn_sequences[ci]):
                segmented_sig = np.zeros(sig_len)
                for k in segment[0]:
                    segmented_sig[:-k if k else sig_len] -= channel[k:]
                for k in segment[1]:
                    segmented_sig[:-k if k else sig_len] += channel[k:]
                segmented_sig *= self.segment_scalars[si]
                output_sig[:, ci] += segmented_sig
        dsp.rms_normalize(input_sig, output_sig)
        return output_sig

    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        """Perform a velvet noise decorrelation on input_sig.

        As of right now only supports stereo decorrelation (i.e. no quad-channel, 5.1, 7.1, etc. support)
        This method will perform an optimized velvet noise convolution for each channel
        to generate the side channel content.
        Then it will interpolate between the mid (dry) and side (wet) channels at self.width.
        Lastly, this method will apply a delay to the mid and left channels, and return the result.

        Parameters
        ----------
        input_sig : np.ndarray
            The mono or stereo input signal to upmix or decorrelate.

        Returns
        -------
        output_sig : np.ndarray
            The stereoized, decorrelated output signal.

        """
        # convert to 32 bit floating point, if it isn't already
        input_sig = dsp.to_float32(input_sig)
        # normalize so that the data ranges from -1 to 1
        dsp.peak_normalize(input_sig)
        # if the input is mono, then duplicate it to stereo before processing
        if input_sig.ndim == 1:
            input_sig = dsp.mono_to_stereo(input_sig)
        output_sig = self.convolve(input_sig)
        output_sig = dsp.encode_signal_to_side_channel(input_sig, output_sig)
        output_sig = dsp.apply_stereo_width(output_sig, self.width)
        return output_sig

    @property
    def density(self) -> float:
        """The average density in impulses per second."""
        return self.num_impulses / self.duration

    @property
    def FIR(self) -> np.ndarray:
        """Return the finite impulse response as a numpy array.

        Returns
        -------
        fir : np.ndarray
            The finite impulse response of the filters as a numpy array of shape (filter_len, num_outs).

        """
        filter_len = int(self.duration * self.fs)
        fir = np.zeros((filter_len, self.num_outs))
        for ci in range(self.num_outs):
            for si, segment in enumerate(self._vn_sequences[ci]):
                for k in segment[0]:
                    fir[k, ci] = -self.segment_scalars[si]
                for k in segment[1]:
                    fir[k, ci] = self.segment_scalars[si]
        return fir

    def regenerate(self) -> None:
        """Regenerate the velvet noise sequences."""
        self._vn_sequences = self._generate()

    def _generate(self) -> List[List[List[List[int]]]]:
        """Generate a velvet noise finite impulse response filter to convolve with an input signal.

        To avoid audible smearing of transients the following are optionally applied:
            - A segmented decay envelope.
            - Logarithmic impulse distribution.
        Segmented decay is preferred to exponential decay because it uses
        less multiplication, and produces satisfactory results.

        Returns
        -------
        velvet_noise : List[List[List[List[int]]]]
            see self._vn_sequences in __init__

        """
        if self.seed is not None:
            np.random.seed(self.seed)
        velvet_noise = []
        sequence_len = int(round(self.fs * self.duration))
        # average spacing between two impulses (grid size)
        grid_size = self.fs / self.density
        #num_impulses = int(sequence_len / grid_size)
        # coefficient values for segmented decay can be set manually
        num_segments = len(self.segment_scalars)
        # calculate the grid size between each impulse logarithmically
        # use a function to apply this for each impulse index
        def log_grid_size(m): return pow(10, m / self.num_impulses)
        # the calculated intervals between impulses
        intervals = np.array([log_grid_size(m) for m in range(self.num_impulses)])
        sum_intervals = np.sum(intervals)
        for ch in range(self.num_outs):
            r1 = np.random.uniform(low=0, high=1, size=self.num_impulses)
            r2 = np.random.uniform(low=0, high=1, size=self.num_impulses)
            sign = (2 * np.round(r1)) - 1
            segments = [[[], []] for _ in self.segment_scalars]
            fir_index = 0 # location of the impulse in the velvet noise sequence
            for m in range(self.num_impulses):
                if self.log_distribution:
                    fir_index = int(np.round(r2[m] * (log_grid_size(m) - 1)) \
                        + np.sum(intervals[:m]) * (sequence_len / sum_intervals))
                else:
                    fir_index = int(np.round(m * grid_size + r2[m] * (grid_size - 1)))
                segment_index = int(m / (self.num_impulses / num_segments))
                sign_index = int((sign[m] + 1) / 2)
                segments[segment_index][sign_index].append(fir_index)
            # filter out unused segments and append to list
            velvet_noise.append(list(filter(None, segments)))
        return velvet_noise

def main():
    """Example usage."""
    import scipy.io.wavfile as wavfile
    fs, sig_float32 = wavfile.read("audio/guitar.wav")
    decorrelators = (
        VelvetNoise(fs=fs, duration=0.03, num_impulses=30, width=1.0),
        HaasEffect(0.0197, fs=fs, channel=1, mode='LR'),
        HaasEffect(0.0096, fs=fs, channel=1, mode='MS'))
    chain = SignalChain(decorrelators, fs, 1, 2)
    output_sig = chain(sig_float32)
    wavfile.write('audio/guitar_dec.wav', fs, output_sig)

if __name__ == '__main__':
    main()
