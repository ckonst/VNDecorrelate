from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

EPSILON: float = 1e-10

IDENTITY_ENVELOPE: tuple[float] = (1.0,)


class NormalizeMode(StrEnum):
    STEREO = 'stereo'
    DUAL_MONO = 'dual_mono'


def apply_stereo_width(input_signal: NDArray, width: float) -> None:
    """Return ``input_signal`` with the Mid-Side balance interpolated at width.

    ``input_signal`` MUST be a Left-Right stereo signal of shape (n, 2).

    Parameters
    ----------
    input_signal : NDArray
        The original signal.
    width : float
        The percentage to scale the side channel by [0.0, 1.0].

    """
    LR_to_MS(input_signal)
    input_signal[:, 0] *= 1.0 - width
    input_signal[:, 1] *= width
    MS_to_LR(input_signal)


def encode_signal_to_side_channel(
    input_signal: NDArray, decorrelated_signal: NDArray
) -> None:
    """Encodes ``decorrelated_signal`` in-place to be the side channel of ``input_signal``.

    ``input_signal`` and decorrelated_signal MUST be a Left-Right stereo signal of shape (n, 2).
    Assumes that ``input_signal`` is a mono signal duplicated to stereo,
    both channels of ``decorrelated_signal`` will be overwritten.

    Parameters
    ----------
    input_signal : NDArray
        The original signal.
    decorrelated_signal : NDArray
        The decorrelated signal.

    """
    check_stereo(input_signal)
    check_stereo(decorrelated_signal)
    M = np.sum(input_signal, axis=1)
    # diff does col 1 - col 0, so swap columns to get L - R
    S = np.diff(decorrelated_signal[:, [1, 0]], axis=1).squeeze() * 0.5
    decorrelated_signal[:, 0] = (M + S) * 0.5
    decorrelated_signal[:, 1] = (M - S) * 0.5


def to_float32(input_signal: NDArray) -> NDArray[np.float32]:
    """Return ``input_signal`` as an array of 32 bit floats."""
    return input_signal.astype(np.float32, copy=False)


def peak_normalize(
    input_signal: NDArray,
    mode: NormalizeMode = NormalizeMode.DUAL_MONO,
    epsilon: float = EPSILON,
) -> None:
    """Normalize ``input_signal`` in-place to ``[-1, 1]`` using the calculated peaks."""

    match (input_signal.ndim, mode):
        case (1, _) | (_, NormalizeMode.STEREO):
            input_axis = None
        case (_, NormalizeMode.DUAL_MONO):
            input_axis = 0

    input_signal *= 1.0 / (np.max(np.abs(input_signal), axis=input_axis) + epsilon)


def rms_normalize(
    input_signal: NDArray,
    output_signal: NDArray,
    mode: NormalizeMode = NormalizeMode.DUAL_MONO,
    epsilon: float = EPSILON,
) -> None:
    """Normalize ``output_signal`` in-place to the rms value of ``input_signal``."""

    match (input_signal.ndim, mode):
        case (1, _) | (_, NormalizeMode.STEREO):
            input_axis = None
        case (_, NormalizeMode.DUAL_MONO):
            input_axis = 0

    match (output_signal.ndim, mode):
        case (1, _) | (_, NormalizeMode.STEREO):
            output_axis = None
        case (_, NormalizeMode.DUAL_MONO):
            output_axis = 0

    output_signal *= np.sqrt(
        np.mean(np.square(input_signal), axis=input_axis)
    ) / np.sqrt(np.mean(np.square(output_signal), axis=output_axis) + epsilon)


def mono_to_stereo(input_signal: NDArray) -> NDArray:
    """Convert a mono signal of shape ``(n,)`` to a stereo signal of shape ``(n, 2)``."""
    check_mono(input_signal)
    return np.column_stack((input_signal, input_signal))


def stereo_to_mono(input_signal: NDArray) -> NDArray:
    """Convert a stereo signal of shape ``(n, 2)`` to a mono signal of shape ``(n,)``."""
    check_stereo(input_signal)
    return np.sum(input_signal, axis=1) * 0.5


def LR_to_MS(input_signal: NDArray) -> None:
    """Given a Left-Right stereo signal, convert it to Mid-Side in-place.

    Converts LR to MS with the formula:
        M = (L + R) / 2
        S = (L - R) / 2
    Requires L as channel 0 and R as channel 1.
    Encodes M into channel 0 and S into channel 1.

    Parameters
    ----------
    input_signal : NDArray
        The original LR stereo signal.

    """
    check_stereo(input_signal)
    M = np.sum(input_signal, axis=1) * 0.5
    # diff computes col 1 - col 0, so swap columns to get M - S
    S = np.diff(input_signal[:, [1, 0]], axis=1).squeeze() * 0.5
    input_signal[:, 0] = M
    input_signal[:, 1] = S


def MS_to_LR(input_signal: NDArray) -> None:
    """Given a Mid-Side stereo signal, convert it to Left-Right in-place.

    Converts MS to LR with the formula:
        L = M + S
        R = M - S
    Requires M as channel 0 and S as channel 1.
    Encodes L as channel 0 and R as channel 1.

    Parameters
    ----------
    input_signal : NDArray
        The original MS stereo signal.

    """
    check_stereo(input_signal)
    L = np.sum(input_signal, axis=1)
    # diff does col 1 - col 0, so swap columns to get M - S
    R = np.diff(input_signal[:, [1, 0]], axis=1).squeeze()
    input_signal[:, 0] = L
    input_signal[:, 1] = R


def generate_log_distribution(strength: float, size: int) -> NDArray:
    """Generate an exponentially increasing distribution array.

    Parameters
    ----------
    strength : float
        Controls how strongly the values increase towards the end of the
        array. ``0.0`` produces a uniform distribution, ``1.0`` produces
        a strong (logarithmic) increase.
    size : int
        Number of intervals; the returned array has length ``size + 1``.

    Returns
    -------
    ndarray
        Array of length ``size + 1`` containing exponentially increasing
        values parameterized by ``strength``.

    Notes
    -----
    The implementation scales and normalizes the exponential curve so that
    the distribution varies smoothly between uniform and strongly
    concentrated shapes as ``strength`` moves from ``0.0`` to ``1.0``.
    """
    return (
        (10.0 ** (2.0 * strength * (np.arange(size + 1.0) / size)))
        /
        # Applying the strength term to the denominator helps with the uniform density case
        # where ``strength = 0.0`` by reducing the normalization constant to 1.0
        # to generate a sequence that increases linearly, while still parameterized by strength.
        (100.0 * ((1.0 + (strength * 99.0)) / 100.0))
    )


def apply_log_distribution(
    randoms: NDArray,
    log_distribution: NDArray,
    log_impulse_intervals: NDArray,
    jitter: float,
) -> NDArray[np.int32]:
    """Compute randomized impulse positions using a logarithmic distribution.

    Parameters
    ----------
    randoms : ndarray
        Uniform random values in ``[0, 1]``; shape must match
        ``log_distribution`` and ``log_impulse_intervals``.
    log_distribution : ndarray
        Precomputed distribution weights from
        :func:``generate_log_distribution`` describing how density varies
        across the filter.
    log_impulse_intervals : ndarray
        Cumulative impulse interval positions corresponding to the
        ``log_distribution`` values.
    jitter : float
        Scale applied to ``log_distribution`` that controls how much the
        random offsets can perturb each impulse position. A value of ``0.0`` means no jitter.
        Typical usage is to set this to the average impulse interval (samples per impulse).

    Returns
    -------
    ndarray of int32
        Rounded sample indices for each randomized impulse position,
        suitable for indexing the FIR.

    Notes
    -----
    The function computes positions as ``round(randoms * np.fmax(0.0, log_distribution *
    jitter - 1) + log_impulse_intervals)`` and casts the result to
    ``np.int32``.

    ``np.fmax`` is used to ensure that the jitter does not cause negative scaling of the random values,
    which could place impulses at the end of the filter instead of the beginning.

    The random values are scaled by the jitter-modified distribution to
    allow for more or less deviation from the base logarithmic positions,
    while ensuring that the impulses remain distributed according to the logarithmic density profile.
    """
    return np.round(
        randoms * np.fmax(0.0, log_distribution * jitter - 1) + log_impulse_intervals
    ).astype(np.int32)


def uniform_density(
    randoms: NDArray,
    impulse_indexes: NDArray,
    impulse_interval: float,
) -> NDArray[np.int32]:
    """Return randomized impulse positions in an FIR with uniform density.

    Parameters
    ----------
    randoms : ndarray
        Uniform random values in the interval ``[0, 1]``, with the same shape
        as ``impulse_indexes``.
    impulse_indexes : ndarray
        Precomputed impulse indices (for example ``np.arange(n_impulses)``),
        where the first impulse is at index ``0``.
    impulse_interval : float
        Average interval between impulses (filter length divided by the
        number of impulses).

    Returns
    -------
    positions : ndarray of int32
        Rounded sample indices for each impulse, suitable for indexing the
        FIR. The array has dtype ``np.int32``.

    Notes
    -----
    The random values are scaled by ``impulse_interval`` so that impulses
    remain uniformly distributed across the filter while each position may
    be shifted by up to ``impulse_interval - 1`` samples.
    """
    return np.round(
        impulse_indexes * impulse_interval + randoms * (impulse_interval - 1)
    ).astype(np.int32)


def check_mono(input_signal: NDArray) -> None:
    """If ``input_signal`` is not a mono signal, raise an error."""
    if input_signal.ndim != 1:
        raise ValueError(
            f'Input shape invalid: Expected shape (num samples,), but got shape {input_signal.shape}.'
        )


def check_stereo(input_signal: NDArray) -> None:
    """If ``input_signal`` is not a stereo signal, raise an error."""
    if input_signal.ndim != 2 or input_signal.shape[1] != 2:
        raise ValueError(
            f'Input shape invalid: Expected shape (num samples, 2), but got shape {input_signal.shape}.'
        )


def check_equal_length(x: NDArray, y: NDArray, dim: int = 0) -> None:
    """If ``x`` and ``y`` do not have equal length for the specified dimension, raise an error."""
    if x.shape[dim] != y.shape[dim]:
        raise ValueError(
            f'Input length mismatch: Expected signals of equal length, but got lengths {x.shape[dim]} and {y.shape[dim]} for dimension {dim}.'
        )


def cross_correlogram(
    x: NDArray,
    y: NDArray,
    sample_rate_hz: int = 44100,
    max_lag_seconds: int = 0.02,
    window_size_seconds: float = 0.02,
    stride_seconds: float = 0.01,
    epsilon: float = EPSILON,
) -> NDArray:
    """Compute the cross-correlogram between ``x`` and ``y``."""
    check_mono(x)
    check_mono(y)
    check_equal_length(x, y)
    x = to_float32(x)
    y = to_float32(y)

    window_size_samples = int(window_size_seconds * sample_rate_hz)
    stride_samples = int(stride_seconds * sample_rate_hz)
    max_lag_samples = int(max_lag_seconds * sample_rate_hz)
    num_lags = 2 * max_lag_samples + 1

    window_indexes = np.arange(0, len(x) - window_size_samples + 1, stride_samples)

    cross_correlogram: NDArray = np.zeros(
        (len(window_indexes), num_lags), dtype=np.float32
    )

    for i, start in enumerate(window_indexes):
        xnk = x[start : start + window_size_samples]
        ynkt = y[start : start + window_size_samples]

        correlation = (
            np.correlate(xnk, ynkt, mode='full')
            / (
                np.sqrt(
                    np.dot(xnk, xnk) * np.dot(ynkt, ynkt)
                )  # normalize by energy of both signals
                + epsilon  # Prevent division by zero
            )
        )[:num_lags]

        cross_correlogram[i, : correlation.shape[0]] = correlation

    return cross_correlogram


def sine_sweep(
    start_freq_hz: float,
    end_freq_hz: float,
    duration_seconds: float,
    sample_rate_hz: int = 44100,
) -> NDArray:
    """Generate a sine sweep signal from ``start_freq_hz`` to ``end_freq_hz`` over ``duration_seconds``."""
    t = np.linspace(
        0, duration_seconds, int(sample_rate_hz * duration_seconds), endpoint=False
    )
    k = np.log(end_freq_hz / start_freq_hz) / duration_seconds
    sine_sweep = np.sin(2 * np.pi * start_freq_hz * (np.exp(k * t) - 1) / k)
    return sine_sweep.astype(np.float32)


def polar_coordinates(
    left: NDArray, right: NDArray, normalize: bool = True
) -> tuple[NDArray, NDArray, NDArray]:
    """Return each sample of the ``left`` and ``right`` channels as polar coordinates with amplitude weights: ``(radii, thetas, weights)``."""
    # atan2 gives angle in radians; the (L-R, L+R) formulation
    # naturally spans [-π/2, π/2] (the 180° stereo field)
    thetas = np.arctan2(left - right, left + right)  # radians, [-π/2, +π/2]
    radii = np.sqrt(left**2 + right**2)  # magnitude [0, √2]
    weights = radii / (radii.sum() + EPSILON)  # amplitude-weights

    if normalize:
        radii /= radii.max() + EPSILON

    return radii, thetas, weights


def exponential_decay(t: float, k: float = 2) -> float:
    """evaluate :math:``e⁻ᵏᵗ`` for a given time ``t`` and decay rate ``k``."""
    return np.e ** (-k * t)


def generate_decay_envelope(
    num_segments: int, segment_position: float
) -> tuple[float, ...]:
    """Generate a decay envelope with ``num_segments`` segments, where the intra-segment sample location is shifted by ``segment_position``.

    Parameters
    ----------
    num_segments : int
        The number of segments to generate.

    segment_position : float
        A percent ``[0.0, 1.0]`` to shift the intra-segment sample location to.

    Returns
    -------
    tuple[float, ...]:
        A tuple of ``num_segments`` floats that define the decay envelope.
    """
    return tuple(
        exponential_decay((t / num_segments) + (segment_position * 1 / num_segments))
        for t in range(num_segments)
    )
