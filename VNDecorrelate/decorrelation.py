from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterator, Protocol, Self, Sequence

import numpy as np
from numpy.typing import NDArray

from VNDecorrelate.utils import dsp
from VNDecorrelate.utils.dsp import log_distribution, log_grid_size, uniform_density

# ----------------------------------------------------------------------------
#
# Abstract Classes and Protocols
#
# ----------------------------------------------------------------------------


@dataclass
class SignalProcessor(Protocol):
    sample_rate_hz: int
    num_ins: int
    num_outs: int

    def __call__(self, input_sig: NDArray) -> NDArray:
        raise NotImplementedError


@dataclass
class Decorrelator(ABC, SignalProcessor):
    """An abstract base class for a Decorrelator."""

    num_ins: int = 2
    num_outs: int = 2
    width: float | None = None

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

_LazyDecorrelator = Callable[[], Decorrelator] | Decorrelator


@dataclass(kw_only=True, slots=True)
class SignalChain(SignalProcessor):
    """A class for building a signal chain of cascading decorrelators."""

    num_ins: int = 2
    num_outs: int = 2
    lazy: bool = True  # If True, wait until the first __call__ to initialize the decorrelators, unless _hot is True

    _hot: bool = False  # If True, bypass lazy loading for every subsequent call to _add_decorrelator
    _decorrelators: list[_LazyDecorrelator] | None = None

    def __post_init__(self) -> None:
        if self._decorrelators is not None:
            raise TypeError(
                'Cannot supply decorrelators directly, use velvet_noise or haas_effect.'
            )
        self._decorrelators = []

        if not self.lazy:
            self._hot = True

    def velvet_noise(self, **kwargs) -> Self:
        self._add_decorrelator(VelvetNoise, **kwargs)
        return self

    def haas_effect(self, **kwargs) -> Self:
        self._add_decorrelator(HaasEffect, **kwargs)
        return self

    def _add_decorrelator(self, cls: type[Decorrelator], **kwargs) -> None:
        kwargs = self._validate(cls, **kwargs)
        match self._decorrelators:
            case []:
                self._decorrelators = [
                    lambda _: cls(
                        sample_rate_hz=self.sample_rate_hz,
                        num_ins=self.num_ins,
                        num_outs=self.num_outs,
                        **kwargs,
                    ),
                ]
            case [*_, _]:
                self._decorrelators = [
                    *self._decorrelators,
                    lambda i: cls(
                        sample_rate_hz=self.sample_rate_hz,
                        num_ins=self._decorrelators[i].num_outs,
                        num_outs=kwargs.get('num_outs', self.num_outs),
                        **kwargs,
                    ),
                ]
        if self._hot:
            self._decorrelators = [
                *self._decorrelators[:-1],
                self._decorrelators[-1](),
            ]  # Construct the decorrelator immediately if we're hot loading.

    def _validate(
        self, cls, *, sample_rate_hz: int | None = None, **kwargs
    ) -> dict[str, Any]:
        if sample_rate_hz is not None and sample_rate_hz != self.sample_rate_hz:
            raise TypeError(
                f'{sample_rate_hz=} was supplied to {cls} but differs from the sample rate of the enclosing SignalChain ({self.sample_rate_hz})'
            )
        return kwargs

    def __call__(self, input_sig: NDArray) -> NDArray:
        """Decorrelate input_sig with all decorrelators in this SignalChain."""
        self._init_decorrelators()
        cascaded_sig = input_sig
        for decorrelator in self._decorrelators:
            cascaded_sig = decorrelator(input_sig)
            input_sig = cascaded_sig
        return cascaded_sig

    def _init_decorrelators(self) -> None:
        if self._hot:
            return

        for i, decorralator in enumerate(self._decorrelators):
            self._decorrelators[i] = decorralator(i - 1)

        self._hot = True


# ----------------------------------------------------------------------------
#
# Haas Effect Decorrelator
#
# ----------------------------------------------------------------------------


class HaasEffectMode(StrEnum):
    LR = 'LR'  # Left-Right
    MS = 'MS'  # Mid-Side


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

    def decorrelate(self, input_sig: NDArray) -> NDArray:
        """Perform a Haas Effect decorrelation on input_sig."""
        input_sig = dsp.to_float32(input_sig)
        output_sig = self.haas_delay(input_sig)

        if self.width is not None:
            output_sig = dsp.apply_stereo_width(output_sig, self.width)

        dsp.rms_normalize(input_sig, output_sig)

        return output_sig

    def haas_delay(self, input_sig: NDArray) -> NDArray:
        """Return a stereo signal where the specified channel is delayed by delay_time."""
        output_sig = input_sig
        delay_len_samples = round(self.delay_time_seconds * self.sample_rate_hz)
        mono = False

        if input_sig.ndim == 1:
            mono = True
            output_sig = dsp.mono_to_stereo(input_sig)

        if self.mode == HaasEffectMode.MS:
            if mono:  # sides will be silent
                mids = dsp.stereo_to_mono(output_sig)
                # this is technically incorrect, but we fix it later.
                sides = mids
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

        location = (
            (dry_channel, wet_channel)
            if self.delayed_channel
            else (wet_channel, dry_channel)
        )
        output_sig = np.column_stack(location)

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


@dataclass
class _VelvetNoiseSegment:
    """Dataclass for storing indexes to nonzero samples of Velvet Noise."""

    negative_impulse_indexes: Sequence[int] = ()
    positive_impulse_indexes: Sequence[int] = ()

    def __iter__(self) -> Iterator[tuple[Sequence[int], str]]:
        """Return an iterator containing a pair of the positive/negative impulse indexes, and the method name to use for the optimized convolution."""
        return iter(
            (
                (self.negative_impulse_indexes, '__isub__'),
                (self.positive_impulse_indexes, '__iadd__'),
            )
        )

    def __getitem__(self, key: int) -> Sequence[int]:
        if key == 0:
            return self.negative_impulse_indexes
        if key == 1:
            return self.positive_impulse_indexes
        raise ValueError('Invalid key')

    def __setitem__(self, key: int, value: Sequence[int]) -> Sequence[int]:
        if key == 0:
            self.negative_impulse_indexes = value
            return
        if key == 1:
            self.positive_impulse_indexes = value
            return
        raise ValueError('Invalid key')


@dataclass
class _VelvetNoiseSequence:
    """Dataclass for storing Segments of Velvet Noise for each output channel."""

    output_channels: Sequence[Sequence[_VelvetNoiseSegment]]

    def __iter__(self) -> Iterator[Sequence[_VelvetNoiseSegment]]:
        return iter(self.output_channels)

    def __getitem__(self, key: int) -> Sequence[_VelvetNoiseSegment]:
        return self.output_channels[key]


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
        use_log_distribution : bool
            Whether to distribute impulses logarithmically.
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
        It maps each channel to a list of equal length segments, determined by segment_envelope.
        Each segment is then split into lists of negative and positive impulses at indices 0 and 1 respectively.
        """
        self._vn_sequence: _VelvetNoiseSequence = self._generate()

    def convolve(self, input_sig: NDArray) -> NDArray:
        """Perform the latency-free convolution of the velvet noise filters onto each channel of a signal."""
        sig_len = len(input_sig)
        segment_buffer = np.zeros(sig_len, dtype=np.float32)
        output_sig = np.zeros((sig_len, self.num_outs), dtype=np.float32)

        for channel_index, channel_segments in enumerate(self._vn_sequence):
            for segment_index, segment in enumerate(channel_segments):
                for signed_indexes, operator in segment:
                    for impulse_index in signed_indexes:
                        getattr(
                            segment_buffer[
                                : -impulse_index or sig_len
                            ],  # Map impulse_index 0 to sig_len to conform to python indexing rules.
                            operator,
                        )(input_sig[:, channel_index][impulse_index:])
                segment_buffer *= self.segment_envelope[segment_index]
                output_sig[:, channel_index] += segment_buffer
                segment_buffer *= 0
        return output_sig

    def decorrelate(self, input_sig: NDArray) -> NDArray:
        """Perform a velvet noise decorrelation on input_sig.

        As of right now only supports stereo decorrelation (i.e. no quad-channel, 5.1, 7.1, etc. support)
        This method will perform an optimized velvet noise convolution for each channel to generate the side channel content.
        Then it will interpolate between the mid (dry) and side (wet) channels at self.width.
        Lastly, the output signal is normalized by the RMS of the input signal.

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
        filter_len = int(self.duration * self.sample_rate_hz)
        fir = np.zeros((filter_len, self.num_outs))
        for channel_index, channel_segments in enumerate(self._vn_sequence):
            for segment_index, segment in enumerate(channel_segments):
                for signed_indexes, operator in segment:
                    for impulse_index in signed_indexes:
                        fir[impulse_index, channel_index] = getattr(
                            fir[impulse_index, channel_index],
                            operator.replace('i', ''),  # ugh
                        )(self.segment_envelope[segment_index])
        return fir

    def _generate(self) -> _VelvetNoiseSequence:
        """Generate a velvet noise finite impulse response filter to convolve with an input signal.

        To avoid audible smearing of transients the following are optionally applied:
            - A segmented decay envelope.
            - Logarithmic impulse distribution.
        Segmented decay is preferred to exponential decay because it uses
        less multiplication, and produces satisfactory results.

        """
        if self.seed is not None:
            np.random.seed(self.seed)

        velvet_noise = _VelvetNoiseSequence(output_channels=())
        sequence_len = int(round(self.sample_rate_hz * self.duration))
        grid_size = (
            self.sample_rate_hz / self.density
        )  # average spacing between two impulses (samples per impulse)
        num_segments = len(self.segment_envelope)

        intervals = np.array(
            [
                log_grid_size(impulse_index, self.num_impulses)
                for impulse_index in range(self.num_impulses)
            ]
        )
        sum_intervals = np.sum(intervals)

        for _ in range(self.num_outs):
            impulse_sign_rng = np.random.uniform(low=0, high=1, size=self.num_impulses)
            impulse_offset_rng = np.random.uniform(
                low=0, high=1, size=self.num_impulses
            )
            sign = (2 * np.round(impulse_sign_rng)) - 1
            segments = tuple(_VelvetNoiseSegment() for _ in self.segment_envelope)
            fir_index = 0  # location of the impulse in the velvet noise sequence
            for impulse_index in range(self.num_impulses):
                fir_index = (
                    log_distribution(
                        impulse_index,
                        impulse_offset_rng[impulse_index],
                        intervals,
                        sum_intervals,
                        sequence_len,
                        self.num_impulses,
                    )
                    if self.use_log_distribution
                    else uniform_density(impulse_index, impulse_offset_rng, grid_size)
                )
                segment_index = int(impulse_index / (self.num_impulses / num_segments))
                sign_index = int((sign[impulse_index] + 1) / 2)
                segments[segment_index][sign_index] = (
                    *segments[segment_index][sign_index],
                    fir_index,
                )

            # filter out unused segments and append to list
            velvet_noise.output_channels = (
                *velvet_noise.output_channels,
                tuple(filter(None, segments)),
            )
        return velvet_noise


if __name__ == '__main__':
    import scipy.io.wavfile as wavfile

    fs, guitar = wavfile.read('audio/guitar.wav')
    VelvetNoise(sample_rate_hz=fs)(guitar)
