from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any, Callable, Iterator, Protocol, Self, Sequence

import numpy as np
from numpy.typing import NDArray

from vndecorrelate.utils.dsp import (
    IDENTITY_ENVELOPE,
    LR_to_MS,
    MS_to_LR,
    apply_log_distribution,
    apply_stereo_width,
    check_equal_length,
    encode_signal_to_side_channel,
    generate_log_distribution,
    mono_to_stereo,
    rms_normalize,
    to_float32,
)

# ----------------------------------------------------------------------------
#
# Abstract Classes and Protocols
#
# ----------------------------------------------------------------------------


@dataclass
class SignalProcessor(Protocol):
    sample_rate_hz: int
    num_outs: int

    def __call__(self, input_signal: NDArray) -> NDArray:
        raise NotImplementedError


@dataclass
class Decorrelator(ABC, SignalProcessor):
    """An abstract base class for a Decorrelator."""

    num_outs: int = 2
    width: float | None = None

    @abstractmethod
    def decorrelate(self, input_signal: NDArray) -> NDArray:
        """Main decorrelation function that must be implemented in subclasses."""
        pass

    def __call__(self, input_signal: NDArray) -> NDArray:
        """Alternative way of calling decorrelate."""
        return self.decorrelate(input_signal)


class StatelessDecorrelator(Protocol):
    def __call__(self, input_signal: NDArray, **kwargs) -> NDArray:
        raise NotImplementedError


# ----------------------------------------------------------------------------
#
# Signal Chain Class
#
# ----------------------------------------------------------------------------

type _LazyDecorrelator = Callable[[], Decorrelator] | Decorrelator


@dataclass(kw_only=True, slots=True)
class SignalChain(SignalProcessor):
    """A class for building a signal chain of cascading decorrelators."""

    num_outs: int = 2
    lazy: bool = True  # If True, wait until the first __call__ to initialize the decorrelators, unless _hot is True

    _hot: bool = False  # If True, bypass lazy loading for every subsequent call to _add_decorrelator
    _decorrelators: list[_LazyDecorrelator] | None = None

    def __post_init__(self) -> None:
        if self._decorrelators is not None:
            raise TypeError(
                'Cannot supply decorrelators directly, use `SignalChain.velvet_noise`,'
                ' `SignalChain.haas_effect`, `SignalChain.white_noise`, or `SignalChain.stateless`.'
            )
        self._decorrelators = []

        if not self.lazy:
            self._hot = True

    def haas_effect(self, **kwargs) -> Self:
        self._add_decorrelator(HaasEffect, **kwargs)
        return self

    def velvet_noise(self, **kwargs) -> Self:
        self._add_decorrelator(VelvetNoise, **kwargs)
        return self

    def white_noise(self, **kwargs) -> Self:
        self._add_decorrelator(WhiteNoise, **kwargs)
        return self

    def stateless(self, function: StatelessDecorrelator, *args, **kwargs):
        self._decorrelators.append(
            partial(function, *args, **kwargs)
            if self._hot
            else lambda: partial(function, *args, **kwargs)
        )
        return self

    def _add_decorrelator(self, cls: type[Decorrelator], **kwargs) -> None:
        kwargs = self._validate(cls, **kwargs)
        self._decorrelators.append(
            cls(
                sample_rate_hz=self.sample_rate_hz,
                num_outs=kwargs.get('num_outs', self.num_outs),
                **kwargs,
            )
            if self._hot
            else lambda: cls(
                sample_rate_hz=self.sample_rate_hz,
                num_outs=kwargs.get('num_outs', self.num_outs),
                **kwargs,
            ),
        )

    def _validate(
        self, cls, *, sample_rate_hz: int | None = None, **kwargs
    ) -> dict[str, Any]:
        if sample_rate_hz is not None and sample_rate_hz != self.sample_rate_hz:
            raise TypeError(
                f'{sample_rate_hz=} was supplied to {cls} but differs from the sample rate of the enclosing SignalChain ({self.sample_rate_hz})'
            )
        return kwargs

    def __call__(self, input_signal: NDArray) -> NDArray:
        """Decorrelate input_signal with all decorrelators in this SignalChain."""
        self._init_decorrelators()
        cascaded_sig = input_signal
        for decorrelator in self._decorrelators:
            cascaded_sig = decorrelator(input_signal)
            input_signal = cascaded_sig
        return cascaded_sig

    def _init_decorrelators(self) -> None:
        if self._hot:
            return

        for i, decorralator in enumerate(self._decorrelators):
            self._decorrelators[i] = decorralator()

        self._hot = True


class DecorrelateMode(StrEnum):
    LR = 'LR'  # Left-Right
    MS = 'MS'  # Mid-Side


# ----------------------------------------------------------------------------
#
# Haas Effect Decorrelator
#
# ----------------------------------------------------------------------------


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
            If set to MS, then the input channels will be converted to Mid-Side as though they were Left-Right, even if they're not.

    """

    delayed_channel: int = 0
    delay_time_seconds: float = 0.02
    mode: DecorrelateMode = DecorrelateMode.LR

    def decorrelate(self, input_signal: NDArray) -> NDArray:
        """Perform a Haas Effect decorrelation on input_signal."""
        input_signal = to_float32(input_signal)
        output_signal = self.haas_delay(input_signal)

        if self.width is not None:
            apply_stereo_width(output_signal, self.width)

        return output_signal

    def haas_delay(self, input_signal: NDArray) -> NDArray:
        """Return a stereo signal where the specified channel is delayed by delay_time."""
        delay_len_samples = round(self.delay_time_seconds * self.sample_rate_hz)
        input_length = len(input_signal)
        output_signal = np.zeros((input_length + delay_len_samples, 2))
        mono = False

        if input_signal.ndim == 1:
            input_signal = mono_to_stereo(input_signal)
            mono = True

        output_signal[:input_length, :] = input_signal

        # side channel will be silent if the signal is mono,
        # since we already duplicated to stereo, we'll interpret that signal as Mid-Side.
        if self.mode == DecorrelateMode.MS and not mono:
            LR_to_MS(output_signal)

        output_signal[:, self.delayed_channel] = np.roll(
            output_signal[:, self.delayed_channel], delay_len_samples, axis=0
        )

        if self.mode == DecorrelateMode.MS:
            MS_to_LR(output_signal)
            if mono:
                # we duplicated the mono channel, here we compensate for it.
                output_signal *= 0.5

        return output_signal


# ----------------------------------------------------------------------------
#
# Velvet Noise Decorrelator
#
# ----------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class _VelvetNoiseSegment:
    """Dataclass for storing indexes to nonzero samples of Velvet Noise."""

    negative_impulse_indexes: Sequence[int] = field(default_factory=list)
    positive_impulse_indexes: Sequence[int] = field(default_factory=list)

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

    def __setitem__(self, key: int, value: Sequence[int]) -> None:
        if key == 0:
            self.negative_impulse_indexes = value
            return
        if key == 1:
            self.positive_impulse_indexes = value
            return
        raise ValueError('Invalid key')


@dataclass(kw_only=True, slots=True)
class _VelvetNoiseSequence:
    """Dataclass for storing Segments of Velvet Noise within a Sequence."""

    segments: Sequence[_VelvetNoiseSegment] = field(default_factory=list)

    @classmethod
    def create(cls, *, num_segments: int) -> Self:
        return cls(segments=[_VelvetNoiseSegment() for _ in range(num_segments)])

    def __iter__(self) -> Iterator[_VelvetNoiseSegment]:
        return iter(self.segments)

    def __getitem__(self, key: int) -> _VelvetNoiseSegment:
        return self.segments[key]

    def __setitem__(self, key: int, value: Sequence[_VelvetNoiseSegment]) -> None:
        self.segments[key] = value


@dataclass(kw_only=True, slots=True)
class _ParallelVelvetNoise:
    """Dataclass for storing Velvet Noise Sequences for each output channel."""

    output_channels: Sequence[_VelvetNoiseSequence] = field(default_factory=list)
    # Store the FIR length to determine if the velvet noise sequence needs to be regenerated in case it changes after initialization.
    fir_length_samples: int

    @property
    def num_outs(self) -> int:
        return len(self.output_channels)

    @property
    def num_impluses(self) -> int:
        return sum(
            len(segment.negative_impulse_indexes)
            + len(segment.positive_impulse_indexes)
            for channel in self.output_channels[
                0:1
            ]  # all channels have the same number of impulses, so we can just check the first one.
            for segment in channel
        )

    def __iter__(self) -> Iterator[_VelvetNoiseSequence]:
        return iter(self.output_channels)

    def __getitem__(self, key: int) -> _VelvetNoiseSequence:
        return self.output_channels[key]

    def __setitem__(self, key: int, value: Sequence[_VelvetNoiseSequence]) -> None:
        self.output_channels[key] = value


@dataclass(kw_only=True, slots=True)
class VelvetNoise(Decorrelator):
    """A `Velvet-Noise Decorrelator <http://www.dafx17.eca.ed.ac.uk/papers/DAFx17_paper_96.pdf>`__ for audio.

    A Velvet Noise Sequence will be generated for each output channel on initialization,
    modifying `duration_seconds`, `num_impulses`, `num_outs`, or `fir_length_samples`
    will trigger a regeneration of the Velvet Noise Sequence on the next call to `velvet_noise`, `convolve`, or `decorrelate`.

    Attributes
    ----------
        duration_seconds : float
            The duration of the velvet noise sequence in seconds.
        num_impulses : int
            The total number of impulses in the velvet noise sequence.
        segment_envelope : Sequence[float]
            The sequence of coefficients for segmented decay, one for each segment.
        log_distribution_strength : float
            How much to distribute impulses logarithmically [0.0, 1.0].
        seed : int
            The seed for the velvet noise generator.
    """

    duration_seconds: float = 0.03
    num_impulses: int = 30
    segment_envelope: Sequence[float] = (0.85, 0.55, 0.35, 0.2)
    log_distribution_strength: float = 1.0
    normalizer: Callable[[NDArray, NDArray], None] | None = rms_normalize
    filtered_channels: tuple[int, ...] = 0, 1
    mode: DecorrelateMode = DecorrelateMode.MS
    seed: int | None = None

    # _velvet_noise maps each channel to a list of equal length segments, determined by segment_envelope.
    # Each segment is then split into lists of negative and positive impulses at indices 0 and 1 respectively.
    _velvet_noise: _ParallelVelvetNoise = ...

    @property
    def velvet_noise(self) -> _ParallelVelvetNoise:
        """The generated velvet noise sequence for each output channel, stored as a dataclass."""
        # Regenerate if duration, num_impulses, num_outs, or fir_length_samples is changed after initialization,
        # since these are the only parameters that affect the velvet noise sequence.
        if (
            self.num_outs != self._velvet_noise.num_outs
            or self.num_impulses != self._velvet_noise.num_impluses
            or self.fir_length_samples != self._velvet_noise.fir_length_samples
        ):
            self._velvet_noise = self._generate()
        return self._velvet_noise

    def __post_init__(self) -> None:
        if self.num_impulses > self.fir_length_samples * 0.5:
            density = self.density
            raise ValueError(
                f'Velvet Noise Filter of length {self.fir_length_samples} with {self.num_impulses} impulses is not sparse. ({density=:.2f})\n'
                '\tnum_impulses must be less than half the FIR length in samples.'
            )
        # If segment_envelope is empty, use a single segment with no decay.
        if not self.segment_envelope:
            self.segment_envelope = IDENTITY_ENVELOPE
        self._velvet_noise = self._generate()

    def convolve(self, input_signal: NDArray) -> NDArray:
        """Perform the optimized convolution of the velvet noise filters onto each channel of a signal."""
        sig_len = len(input_signal)
        segment_buffer = np.zeros(sig_len, dtype=np.float32)
        output_signal = np.zeros((sig_len, self.num_outs), dtype=np.float32)

        for channel_index, channel_segments in enumerate(self.velvet_noise):
            if channel_index not in self.filtered_channels:
                output_signal[:, channel_index] = input_signal[:, channel_index]
                continue
            for segment_index, segment in enumerate(channel_segments):
                for signed_indexes, operator in segment:
                    for impulse_index in signed_indexes:
                        getattr(
                            # Map impulse_index 0 to sig_len to conform to python indexing rules.
                            segment_buffer[: -impulse_index or sig_len],
                            operator,  # either __isub__ or __iadd__
                        )(input_signal[impulse_index:, channel_index])
                if self.segment_envelope != IDENTITY_ENVELOPE:
                    segment_buffer *= self.segment_envelope[segment_index]
                output_signal[:, channel_index] += segment_buffer
                segment_buffer.fill(0)
        return output_signal

    def decorrelate(self, input_signal: NDArray) -> NDArray:
        """Perform a velvet noise decorrelation on input_signal.

        As of right now only supports stereo decorrelation (i.e. no quad-channel, 5.1, 7.1, etc. support)
        This method will perform an optimized velvet noise convolution for each channel to generate the side channel content.
        Then it will interpolate between the mid (dry) and side (wet) channels at self.width.
        Lastly, the output signal is normalized by the RMS of the input signal.

        """
        input_signal = to_float32(input_signal)

        if input_signal.ndim == 1:
            input_signal = mono_to_stereo(input_signal)

        output_signal = self.convolve(input_signal)

        if self.mode == DecorrelateMode.MS:
            encode_signal_to_side_channel(input_signal, output_signal)

        if self.width is not None:
            apply_stereo_width(output_signal, self.width)

        if self.normalizer:
            self.normalizer(input_signal, output_signal)

        return output_signal

    @property
    def density(self) -> float:
        """The average density in impulses per second. Homogeneous across all channels."""
        return self.num_impulses / self.duration_seconds

    @property
    def fir_length_samples(self) -> int:
        """The length of the finite impulse response in samples. Homogeneous across all channels."""
        return int(round(self.sample_rate_hz * self.duration_seconds))

    @property
    def FIR(self) -> NDArray:
        """Return the finite impulse responses (for each channel) as a numpy array of shape (fir_length_samples, num_outs)."""
        fir = np.zeros((self.fir_length_samples, self.num_outs))
        indexes, values = (
            [[] for _ in range(self.num_outs)],
            [[] for _ in range(self.num_outs)],
        )
        for channel_index, channel_segments in enumerate(self.velvet_noise):
            for segment_index, segment in enumerate(channel_segments):
                for (signed_indexes, _), sign in zip(segment, (-1, 1)):
                    for impulse_index in signed_indexes:
                        indexes[channel_index].append(impulse_index)
                        values[channel_index].append(
                            self.segment_envelope[segment_index] * sign
                        )
        np.put_along_axis(fir, np.array(indexes).T, np.array(values).T, 0)
        return fir

    def _generate(self) -> _ParallelVelvetNoise:
        """Generate a velvet noise Finite Impulse Response (FIR) filter to convolve with an input signal.

        To avoid audible smearing of transients the following are optionally applied:
            - A segmented decay envelope.
            - Logarithmic impulse distribution.
        Segmented decay is preferred to exponential decay because it uses
        less multiplication, and produces satisfactory results.

        """
        rng = np.random.default_rng(self.seed)

        velvet_noise = _ParallelVelvetNoise(fir_length_samples=self.fir_length_samples)
        num_segments = len(self.segment_envelope)

        # Array of size (num_impulses + 1) increasing exponentially from [0, 1.0]
        log_distribution = generate_log_distribution(
            self.log_distribution_strength, self.num_impulses
        )

        # Calculate number of samples between each logarithmically distributed impulse:
        # Start by taking the cumulative sum, then shift left by 1 if we have a log distribution strength of 0.0
        # This is done to be consistent with `uniform_density` i.e. we fix an off-by-one error in the uniform distribution case.
        # Lastly, scale the cumulative sum by the length of the filter.
        log_impulse_intervals = np.cumsum(log_distribution)
        if self.log_distribution_strength == 0.0:
            log_impulse_intervals -= 1.0

        log_impulse_intervals *= self.fir_length_samples / log_impulse_intervals[-1]

        impulse_signs = rng.uniform(
            low=0,
            high=1,
            size=(self.num_impulses, self.num_outs),
        )
        impulse_offsets = rng.uniform(
            low=0,
            high=1,
            # Rather than removing the unused last value via copy, just increase the size to match the impulse intervals.
            size=(self.num_impulses + 1, self.num_outs),
        )
        signs = (2 * np.round(impulse_signs)) - 1

        average_impulse_interval = self.sample_rate_hz / self.density

        for channel_index in range(self.num_outs):
            fir_indexes = apply_log_distribution(
                impulse_offsets[:, channel_index],
                log_distribution,
                log_impulse_intervals,
                average_impulse_interval,
            )

            sequence = _VelvetNoiseSequence.create(num_segments=num_segments)

            for impulse_index in range(self.num_impulses):
                segment_index = int(impulse_index / (self.num_impulses / num_segments))
                sign_index = int((signs[impulse_index, channel_index] + 1) / 2)
                sequence[segment_index][sign_index].append(fir_indexes[impulse_index])

            velvet_noise.output_channels.append(sequence)

        return velvet_noise


def generate_velvet_noise(
    *,
    duration_seconds: float,
    num_impulses: int,
    num_outs: int = 2,
    sample_rate_hz: int = 44100,
    segment_envelope: Sequence[float] = (0.85, 0.55, 0.35, 0.2),
    log_distribution_strength: float = 1.0,
    seed: int | None = None,
) -> NDArray:
    """More performant alternative to `VelvetNoise.FIR` for generating a velvet noise FIR filter as an NDArray.

    Generate a velvet noise Finite Impulse Response (FIR) filter to convolve with an input signal.

    To avoid audible smearing of transients the following are optionally applied:
        - A segmented decay envelope.
        - Logarithmic impulse distribution.
    Segmented decay is preferred to exponential decay because it uses
    less multiplication, and produces satisfactory results.

    To convolve after calling this function, you must use `convolve_velvet_noise`.
    For batch processing or long audio files `VelvetNoise` will usually be more performant.

    """
    rng = np.random.default_rng(seed)

    fir_length_samples = int(duration_seconds * sample_rate_hz)

    velvet_noise = np.zeros((fir_length_samples, num_outs), dtype=np.float32)

    # If segment_envelope is empty, use a single segment with no decay.
    if not segment_envelope:
        segment_envelope = (1.0,)
    num_segments = len(segment_envelope)

    # Array of size (num_impulses + 1) increasing exponentially from [0, 1.0]
    log_distribution = generate_log_distribution(
        log_distribution_strength, num_impulses
    )

    # Calculate number of samples between each logarithmically distributed impulse:
    # Start by taking the cumulative sum, then shift left by 1 if we have a log distribution strength of 0.0
    # This is done to be consistent with `uniform_density` i.e. we fix an off-by-one error in the uniform distribution case.
    log_impulse_intervals = np.cumsum(log_distribution)
    if log_distribution_strength == 0.0:
        log_impulse_intervals -= 1.0

    log_impulse_intervals *= fir_length_samples / log_impulse_intervals[-1]

    impulse_signs = rng.uniform(
        low=0,
        high=1,
        size=(num_impulses, num_outs),
    )
    impulse_offsets = rng.uniform(
        low=0,
        high=1,
        # Rather than removing the unused last value via copy, just increase the size to match the impulse intervals.
        size=(num_impulses + 1, num_outs),
    )
    signs = (2 * np.round(impulse_signs)) - 1

    average_impulse_interval = sample_rate_hz / (num_impulses / duration_seconds)

    for channel_index in range(num_outs):
        fir_indexes = apply_log_distribution(
            impulse_offsets[:, channel_index],
            log_distribution,
            log_impulse_intervals,
            average_impulse_interval,
        )

        for impulse_index in range(num_impulses):
            segment_index = int(impulse_index / (num_impulses / num_segments))
            velvet_noise[fir_indexes[impulse_index], channel_index] = (
                signs[impulse_index, channel_index] * segment_envelope[segment_index]
            )

    return velvet_noise


def convolve_velvet_noise(
    input_signal: NDArray, velvet_noise_filters: NDArray
) -> NDArray:
    """Simple, stateless, but less performant alternative to `VelvetNoise.convolve()` for convolving velvet noise filters.
    `input_signal` is an `NDArray` of shape (n, num_channels) and `velvet_noise_filters` is an `NDArray` of shape (m, num_channels).
    A `ValueError` is raised if the number of channels differs between `input_signal` and `velvet_noise_filters`.

    `velvet_noise_filters` may also be assigned from `VelvetNoise.FIR`. If `generate_velvet_noise` was used, then this function must be used for convolution.
    While this convolution is optimized to take advantage of the sparseness of the velvet noise, it is generally slower due to more memory allocations.
    """
    num_channels = 1 if input_signal.ndim == 1 else input_signal.shape[1]

    if num_channels > 1:
        check_equal_length(input_signal, velvet_noise_filters, dim=1)

    sig_len = len(input_signal)

    output_signal = np.zeros(input_signal.shape, dtype=np.float32)

    for channel_index in range(num_channels):
        velvet_noise = velvet_noise_filters[:, channel_index]
        for index, value in zip(
            *np.where(velvet_noise != 0.0),
            velvet_noise[velvet_noise != 0.0],
        ):
            # Map index 0 to sig_len to conform to python indexing rules.
            output_signal[: -index or sig_len, channel_index] += (
                input_signal[index:, channel_index] * value
            )

    return output_signal


# ----------------------------------------------------------------------------
#
# White Noise Decorrelator
#
# ----------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class WhiteNoise(Decorrelator):
    duration_seconds: int = 0.03
    seed: int | None = None

    _white_noise_filter: NDArray = ...

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.white_noise_filter = rng.normal(
            loc=0, scale=1, size=(self.fir_length_samples, self.num_outs)
        )

    def decorrelate(self, input_signal: NDArray) -> NDArray:
        input_signal = to_float32(input_signal)

        if input_signal.ndim == 1:
            input_signal = mono_to_stereo(input_signal)

        sig_len = len(input_signal)

        output_signal = np.zeros((sig_len, self.num_outs), dtype=np.float32)

        for channel_index in range(self.num_outs):
            output_signal[:, channel_index] = np.convolve(
                input_signal[:, channel_index],
                self.white_noise_filter[:, channel_index],
                mode='same',
            )

        if self.width is not None:
            apply_stereo_width(output_signal, self.width)

        rms_normalize(input_signal, output_signal)

        return output_signal

    @property
    def fir_length_samples(self) -> int:
        """The length of the finite impulse response in samples. Homogeneous across all channels."""
        return int(round(self.sample_rate_hz * self.duration_seconds))

    @property
    def FIR(self) -> NDArray:
        """Return the finite impulse responses (for each channel) as a numpy array of shape (fir_length_samples, num_outs)."""
        return self._white_noise_filter
