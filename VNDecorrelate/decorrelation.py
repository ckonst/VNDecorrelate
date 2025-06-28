from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from VNDecorrelate.utils import dsp

# ----------------------------------------------------------------------------
#
# Abstract Decorrelator Class
#
# ----------------------------------------------------------------------------


class SignalProcessor(Protocol):
    def __call__(self, input_sig: NDArray) -> NDArray:
        raise NotImplementedError


@dataclass
class Decorrelator(ABC, SignalProcessor):
    """An abstract base class for a Decorrelator."""

    fs: int = 44100  # sampling frequency
    num_ins: int = 2  # number of input channels
    num_outs: int = 2  # number of output channels
    width: float = None  # width of the decorrelation effect

    @abstractmethod
    def decorrelate(self, input_sig: NDArray) -> NDArray:
        """Main decorrelation function that must be implemented in subclasses."""
        pass

    def __call__(self, input_sig: NDArray) -> NDArray:
        """Alternative way of calling decorrelate."""
        return self.decorrelate(input_sig)


# ----------------------------------------------------------------------------
#
# Signal Chain Class
#
# ----------------------------------------------------------------------------


class SignalChain(SignalProcessor):
    """A simple class for cascading decorrelators."""

    def __init__(self, *decorrelators: Decorrelator, **kwargs):
        """Initialize the same way any other decorrelator would."""
        super().__init__(**kwargs)
        self.decorrelators = decorrelators

    def velvet_noise(self, *args, **kwargs):
        self.decorrelators = (*self.decorrelators, VelvetNoise(*args, **kwargs))
        return self

    def haas_effect(self, *args, **kwargs):
        self.decorrelators = (*self.decorrelators, HaasEffect(*args, **kwargs))
        return self

    def __call__(self, input_sig: NDArray) -> NDArray:
        """Decorrelate input_sig with all decorrelators in this SignalChain.

        Parameters
        ----------
        input_sig : NDArray
            The original signal.

        Returns
        -------
        cascaded_sig : NDArray
            The signal processed by cascading decorrelators.

        """
        for decorrelator in self.decorrelators:
            (cascaded_sig := decorrelator(input_sig))
            input_sig = cascaded_sig
        return cascaded_sig


# ----------------------------------------------------------------------------
#
# Haas Effect Decorrelator
#
# ----------------------------------------------------------------------------


class HaasEffectMode(StrEnum):
    """Either LR (Left-Right) or MS (Mid-Side)"""

    LR = 'LR'
    MS = 'MS'


@dataclass(kw_only=True, slots=True)
class HaasEffect(Decorrelator):
    """A Decorrelator that utilizes the Haas Effect.

    In music, the Haas Effect delays a channel or channels by a small amount to alter the cross correlation between each channel.
    For stereo audio, this can be easily done on either the left or right channel, or on the mid or side channel for different results.

    Left-Right and Mid-Side delay values can be chosen to reduce phase incoherence.
    In general, keeping these values under 20ms helps reduce audible doubling artifacts, which are more noticible in mono.

    Attributes
    ----------
        delayed_channel : int
            Which channel gets delayed. For stereo audio:
                0 is the left channel, 1 is the right channel.
                OR if mode is MS:
                0 is the mid channel, 1 is the side channel.
        delay_time_seconds : float
            The time in seconds to delay the channel by.
        mode : HaasEffectMode
            If set to MS, then the input channels will be converted to Mid-Side As though they were Left-Right, even if they're not!

    """

    delayed_channel: int = 0
    delay_time_seconds: float = 0.02
    mode: HaasEffectMode = HaasEffectMode.LR
    width: float = 0.5

    def decorrelate(self, input_sig: NDArray) -> NDArray:
        """Perform a Haas Effect decorrelation on input_sig.

        Parameters
        ----------
        input_sig : NDArray
            The mono or stereo input signal to upmix or decorrelate.

        Returns
        -------
        output_sig : NDArray
            The stereoized, decorrelated output signal.

        """
        # convert to 32 bit floating point, if it isn't already
        input_sig = dsp.to_float32(input_sig)

        # perform the decorrelation
        output_sig = self.haas_delay(input_sig)

        # adjust the width parameter if present.
        if self.width is not None:
            output_sig = dsp.apply_stereo_width(output_sig, self.width)

        # normalize by rms to match input
        dsp.rms_normalize(input_sig, output_sig)

        return output_sig

    def haas_delay(self, input_sig: NDArray) -> NDArray:
        """Return a stereo signal where the specified channel is delayed by delay_time.

        Parameters
        ----------
        input_sig : NDArray
            The input signal to apply the Haas Effect to.

        Returns
        -------
        output_sig : NDArray
            The wet signal with one channel delayed.

        """
        output_sig = input_sig
        delay_len_samples = round(self.delay_time_seconds * self.fs)
        mono = False

        # if the input was mono, convert it to stereo.
        if input_sig.ndim == 1:
            mono = True
            output_sig = dsp.mono_to_stereo(input_sig)

        # if applicable, convert the left-right signal to a mid-side signal.
        if self.mode == HaasEffectMode.MS:
            if mono:  # sides will be silent
                mids = dsp.stereo_to_mono(output_sig)
                # this is technically incorrect, but we fix it later.
                sides = mids
                # now stack them and store back into audio_sig
                output_sig = np.column_stack((mids, sides))
            else:
                output_sig = dsp.LR_to_MS(output_sig)

        zero_padding_sig = np.zeros(delay_len_samples)
        wet_channel = np.concatenate(
            (zero_padding_sig, output_sig[:, self.delayed_channel])
        )
        dry_channel = np.concatenate(
            (output_sig[:, -(self.delayed_channel - 1)], zero_padding_sig)
        )

        # get the location of the wet and dry channels
        # then put them in a tuple so that they can be stacked
        location = (
            (dry_channel, wet_channel)
            if self.delayed_channel
            else (wet_channel, dry_channel)
        )
        output_sig = np.column_stack(location)

        # convert back to left-right, if we delayed the mid and sides.
        if self.mode == HaasEffectMode.MS:
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


@dataclass(kw_only=True)
class VelvetNoise(Decorrelator):
    """A velvet noise decorrelator for audio.

    See method docstrings for more details.

    Attributes
    ----------
        duration : float
            The duration of the velvet noise sequence in seconds.
        num_impulses : int
            The total number of impulses in the velvet noise sequence.
        segment_envelope : Sequence[float]
            The sequence of coefficients for segmented decay, one for each segment.
        seed : int
            The seed for the velvet noise generator.

    """

    duration: float = 0.03
    num_impulses: int = 30
    segment_envelope: Sequence[float] = (0.85, 0.55, 0.35, 0.2)
    use_log_distribution: bool = True
    seed: int = None

    def __post_init__(self):
        """"""

        """
        _vn_sequences is of shape (num_outs, num_segments, 2, num_positive/num_negative)

        It maps each channel to a list of equal length segments, determined by segment_envelope.
        Each segment is then split into lists of negative and positive impulses at indices 0 and 1 respectively.
        Because of the heterogeneity of the last dimension, this cannot be converted to a numpy array.
        Dimension 2 (of size 2) could be of Tuples, but for simplicity,
        We use Lists since the rest of the sequence must be generated with Lists.
        """
        self._vn_sequences: List[List[List[List[int]]]] = self._generate()

    def convolve(self, input_sig: NDArray) -> NDArray:
        """Perform the convolution of the velvet noise filters onto each channel of a signal.

        We take advantage of the sparse nature of the sequence to perform a latency-free convolution.

        Parameters
        ----------
        input_sig : NDArray
            The input signal to convolve with the generated filter.

        Returns
        -------
        output_sig : NDArray
            The stereoized, decorrelated output signal.
        """
        sig_len = len(input_sig)
        segment_buffer = np.zeros(sig_len)
        output_sig = np.zeros((sig_len, self.num_outs))

        for ci, channel in enumerate(input_sig.T):
            for si, (negative_impulses, postive_impulses) in enumerate(
                self._vn_sequences[ci]
            ):
                segment_buffer *= 0
                for k in negative_impulses:
                    # Map 0 to sig_len to conform to python indexing rules
                    segment_buffer[: -k or sig_len] -= channel[k:]
                for k in postive_impulses:
                    # Map 0 to sig_len to conform to python indexing rules
                    segment_buffer[: -k or sig_len] += channel[k:]
                segment_buffer *= self.segment_envelope[si]
                output_sig[:, ci] += segment_buffer
        return output_sig

    def decorrelate(self, input_sig: NDArray) -> NDArray:
        """Perform a velvet noise decorrelation on input_sig.

        As of right now only supports stereo decorrelation (i.e. no quad-channel, 5.1, 7.1, etc. support)
        This method will perform an optimized velvet noise convolution for each channel to generate the side channel content.
        Then it will interpolate between the mid (dry) and side (wet) channels at self.width.
        Lastly, the output signal is normalized by the RMS of the input signal.

        Parameters
        ----------
        input_sig : NDArray
            The mono or stereo input signal to upmix or decorrelate.

        Returns
        -------
        output_sig : NDArray
            The stereoized, decorrelated output signal.

        """
        # convert to 32 bit floating point, if it isn't already
        input_sig = dsp.to_float32(input_sig)

        # if the input is mono, then duplicate it to stereo before processing
        if input_sig.ndim == 1:
            input_sig = dsp.mono_to_stereo(input_sig)

        output_sig = self.convolve(input_sig)

        output_sig = dsp.encode_signal_to_side_channel(input_sig, output_sig)
        if self.width is not None:
            output_sig = dsp.apply_stereo_width(output_sig, self.width)

        # normalize by rms to match input
        dsp.rms_normalize(input_sig, output_sig)

        return output_sig

    @property
    def density(self) -> float:
        """The average density in impulses per second."""
        return self.num_impulses / self.duration

    @property
    def FIR(self) -> NDArray:
        """Return the finite impulse response as a numpy array.

        Returns
        -------
        fir : NDArray
            The finite impulse response of the filters as a numpy array of shape (filter_len, num_outs).

        """
        filter_len = int(self.duration * self.fs)
        fir = np.zeros((filter_len, self.num_outs))
        for ci in range(self.num_outs):
            for si, (negative_impulses, postive_impulses) in enumerate(
                self._vn_sequences[ci]
            ):
                for k in negative_impulses:
                    fir[k, ci] = -self.segment_envelope[si]
                for k in postive_impulses:
                    fir[k, ci] = self.segment_envelope[si]
        return fir

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
        grid_size = self.fs / self.density  # average spacing between two impulses
        num_segments = len(self.segment_envelope)

        def log_grid_size(m):
            return pow(10, m / self.num_impulses)

        intervals = np.array([log_grid_size(m) for m in range(self.num_impulses)])
        sum_intervals = np.sum(intervals)

        def log_distribution(m, r2):
            return int(
                np.round(r2[m] * (log_grid_size(m) - 1))
                + np.sum(intervals[:m]) * (sequence_len / sum_intervals)
            )

        def uniform_density(m, r2):
            return int(np.round(m * grid_size + r2[m] * (grid_size - 1)))

        for ch in range(self.num_outs):
            r1 = np.random.uniform(low=0, high=1, size=self.num_impulses)
            r2 = np.random.uniform(low=0, high=1, size=self.num_impulses)
            sign = (2 * np.round(r1)) - 1
            segments = [[[], []] for _ in self.segment_envelope]
            fir_index = 0  # location of the impulse in the velvet noise sequence
            for m in range(self.num_impulses):
                fir_index = (
                    log_distribution(m, r2)
                    if self.use_log_distribution
                    else uniform_density(m, r2)
                )
                segment_index = int(m / (self.num_impulses / num_segments))
                sign_index = int((sign[m] + 1) / 2)
                segments[segment_index][sign_index].append(fir_index)

            # filter out unused segments and append to list
            velvet_noise.append(list(filter(None, segments)))
        return velvet_noise
