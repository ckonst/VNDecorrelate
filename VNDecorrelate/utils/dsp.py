from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

EPSILON = 1e-10


class NormalizeMode(StrEnum):
    STEREO = 'stereo'
    DUAL_MONO = 'dual_mono'


def apply_stereo_width(input_signal: NDArray, width: float) -> None:
    """Return `input_signal` with the Mid-Side balance interpolated at width.

    `input_signal` MUST be a Left-Right stereo signal of shape (n, 2).

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
    """Encodes `decorrelated_signal` in-place to be the side channel of `input_signal`.

    `input_signal` and decorrelated_signal MUST be a Left-Right stereo signal of shape (n, 2).
    Assumes that `input_signal` is a mono signal duplicated to stereo,
    both channels of `decorrelated_signal` will be overwritten.

    Parameters
    ----------
    input_signal : NDArray
        The original signal.
    decorrelated_signal : NDArray
        The decorrelated signal.

    Returns
    -------
    NDArray
        The newly-encoded signal.

    """
    check_stereo(input_signal)
    check_stereo(decorrelated_signal)
    M = np.sum(input_signal, axis=1)
    # diff does col 1 - col 0, so swap columns to get L - R
    S = np.diff(decorrelated_signal[:, [1, 0]], axis=1).squeeze() * 0.5
    decorrelated_signal[:, 0] = (M + S) * 0.5
    decorrelated_signal[:, 1] = (M - S) * 0.5


def to_float32(input_signal: NDArray) -> NDArray[np.float32]:
    """Return `input_signal` as an array of 32 bit floats."""
    return input_signal.astype(np.float32, copy=False)


def peak_normalize(
    input_signal: NDArray,
    mode: NormalizeMode = NormalizeMode.DUAL_MONO,
    epsilon: float = EPSILON,
) -> None:
    """Normalize `input_signal` in-place to [-1, 1] using the calculated peaks."""

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
    """Normalize output_signal in-place to the rms value of input_signal."""

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
    """Convert a mono signal of shape (n,) to a stereo signal of shape (n, 2)."""
    check_mono(input_signal)
    return np.column_stack((input_signal, input_signal))


def stereo_to_mono(input_signal: NDArray) -> NDArray:
    """Convert a stereo signal of shape (n, 2) to a mono signal of shape (n,)."""
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
    # diff does col 1 - col 0, so swap columns to get M - S
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


def log_distribution(
    randoms: NDArray,
    log_impulse_intervals: NDArray,
    cumsum_log_impulse_intervals: NDArray,
) -> NDArray:
    """Return the randomized position of the impulse in the FIR, distributing logarithmically towards the start of the filter."""
    return (
        np.round(randoms * (log_impulse_intervals[:-1] - 1))
        + cumsum_log_impulse_intervals
    ).astype(np.int32)


def uniform_density(
    randoms: NDArray,
    impulse_indexes: NDArray,
    impulse_interval: float,
) -> int:
    """Return the randomized position of the impulse in the FIR, preserving a uniform density across the filter."""
    return np.round(
        impulse_indexes * impulse_interval + randoms * (impulse_interval - 1)
    ).astype(np.int32)


def check_mono(input_signal: NDArray) -> None:
    """If the input signal is not a mono signal, raise an error."""
    if input_signal.ndim != 1:
        raise ValueError(
            f'Input shape invalid: Expected shape (num samples,), but got shape {input_signal.shape}.'
        )


def check_stereo(input_signal: NDArray) -> None:
    """If the input signal is not a stereo signal, raise an error."""
    if input_signal.ndim != 2 or input_signal.shape[1] != 2:
        raise ValueError(
            f'Input shape invalid: Expected shape (num samples, 2), but got shape {input_signal.shape}.'
        )


def check_equal_length(x: NDArray, y: NDArray, dim: int = 0) -> None:
    """If the input signals do not have equal length for the specified dimension, raise an error."""
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
    """Compute the cross-correlogram between two mono signals."""
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
    """Generate a sine sweep signal from start_freq_hz to end_freq_hz over duration_seconds."""
    t = np.linspace(
        0, duration_seconds, int(sample_rate_hz * duration_seconds), endpoint=False
    )
    k = np.log(end_freq_hz / start_freq_hz) / duration_seconds
    sine_sweep = np.sin(2 * np.pi * start_freq_hz * (np.exp(k * t) - 1) / k)
    return sine_sweep.astype(np.float32)


def polar_coordinates(
    left: NDArray, right: NDArray, normalize: bool = True
) -> tuple[NDArray, NDArray]:
    """Return each sample of the left and right channels as polar coordinates: r, theta."""
    # atan2 gives angle in radians; the (L-R, L+R) formulation
    # naturally spans -pi/2 ... +pi/2 (the 180° stereo field)
    theta = np.arctan2(left - right, left + right)  # radians, -π/2 to +π/2
    r = np.sqrt(left**2 + right**2)  # magnitude [0, √2]

    if normalize:
        r_max = r.max()
        if r_max > 0:
            r /= r_max

    return r, theta
