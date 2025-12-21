import numpy as np
from numpy.typing import NDArray


def apply_stereo_width(input_sig: NDArray, width: float) -> NDArray:
    """Return input_sig with the Mid-Side balance interpolated at width.

    input_sig MUST be a Left-Right stereo signal of shape (n, 2).

    Parameters
    ----------
    input_sig : NDArray
        The original signal.
    width : float
        The percentage to scale the side channel by [0.0, 1.0].

    Returns
    -------
    NDArray
        The width-adjusted signal.

    """
    input_sig = LR_to_MS(input_sig)
    input_sig[:, 0] *= 1.0 - width
    input_sig[:, 1] *= width
    return MS_to_LR(input_sig)


def encode_signal_to_side_channel(
    input_sig: NDArray, decorrelated_sig: NDArray
) -> NDArray:
    """Encodes decorrelated_sig to be the side channel of input_sig.

    input_sig and decorrelated_sig MUST be a Left-Right stereo signal of shape (n, 2).
    Assumes that input_sig is a mono signal duplicated to stereo,
    i.e. input_sig's side channel will be overwritten.

    Parameters
    ----------
    input_sig : NDArray
        The original signal.
    decorrelated_sig : NDArray
        The output signal.

    Returns
    -------
    NDArray
        The newly-encoded signal.

    """
    check_stereo(input_sig)
    check_stereo(decorrelated_sig)
    L = input_sig[:, 0]
    R = input_sig[:, 1]
    Ld = decorrelated_sig[:, 0]
    Rd = decorrelated_sig[:, 1]
    M = L + R
    S = Ld - Rd
    return np.column_stack(((M + S) * 0.5, ((M - S) * 0.5)))


def to_float32(input_sig: NDArray) -> NDArray[np.float32]:
    """Return input_sig as an array of 32 bit floats."""
    return input_sig.astype(np.float32) if input_sig.dtype != np.float32 else input_sig


def peak_normalize(input_sig: NDArray) -> None:
    """Normalize input_sig in-place to [-1, 1] using the calculated peaks."""
    if (max_abs := np.max(np.abs(input_sig))) != 0.0:
        input_sig *= 1.0 / max_abs


def rms_normalize(input_sig: NDArray, output_sig: NDArray) -> None:
    """Normalize output_sig in-place to the rms value of input_sig."""
    if (rms := np.sqrt(np.mean(np.square(np.sum(output_sig, axis=1))))) != 0.0:
        output_sig *= np.sqrt(np.mean(np.square(input_sig))) / rms


def mono_to_stereo(input_sig: NDArray) -> NDArray:
    """Convert a mono signal of shape (n,) to a stereo signal of shape (n, 2)."""
    check_mono(input_sig)
    return np.column_stack((input_sig, input_sig))


def stereo_to_mono(input_sig: NDArray) -> NDArray:
    """Convert a stereo signal of shape (n, 2) to a mono signal of shape (n,)."""
    check_stereo(input_sig)
    return (input_sig[:, 0] + input_sig[:, 1]) * 0.5


def LR_to_MS(input_sig: NDArray) -> NDArray:
    """Given a Left-Right stereo signal, return its Mid-Side equivalent.

    Converts LR to MS with the formula:
        M = (L + R) / 2
        S = (L - R) / 2
    Requires L as channel 0 and R as channel 1.
    Encodes M into channel 0 and S into channel 1.

    Parameters
    ----------
    input_sig : NDArray
        The original LR stereo signal.

    Returns
    -------
    NDArray
        The output stereo signal in Mid-Side domain.

    """
    check_stereo(input_sig)
    M = (input_sig[:, 0] + input_sig[:, 1]) * 0.5
    S = (input_sig[:, 0] - input_sig[:, 1]) * 0.5
    return np.column_stack((M, S))


def MS_to_LR(input_sig: NDArray) -> NDArray:
    """Given a Mid-Side stereo signal return its Left-Right equivalent.

    Converts MS to LR with the formula:
        L = M + S
        R = M - S
    Requires M as channel 0 and S as channel 1.
    Encodes L as channel 0 and R as channel 1.

    Parameters
    ----------
    input_sig : NDArray
        The original MS stereo signal.

    Returns
    -------
    NDArray
        The output stereo signal in Left-Right domain.

    """
    check_stereo(input_sig)
    L = input_sig[:, 0] + input_sig[:, 1]
    R = input_sig[:, 0] - input_sig[:, 1]
    return np.column_stack((L, R))


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


def check_mono(input_sig: NDArray) -> None:
    """If the input signal is not a mono signal, raise an error."""
    if input_sig.ndim != 1:
        raise ValueError(
            f'Input shape invalid: Expected shape (num samples,), but got shape {input_sig.shape}.'
        )


def check_stereo(input_sig: NDArray) -> None:
    """If the input signal is not a stereo signal, raise an error."""
    if input_sig.ndim != 2 or input_sig.shape[1] != 2:
        raise ValueError(
            f'Input shape invalid: Expected shape (num samples, 2), but got shape {input_sig.shape}.'
        )


def check_equal_length(x: NDArray, y: NDArray) -> None:
    """If the input signals do not have equal length, raise an error."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f'Input length mismatch: Expected signals of equal length, but got lengths {x.shape[0]} and {y.shape[0]}.'
        )


def cross_correlogram(
    x: NDArray,
    y: NDArray,
    sample_rate_hz: int = 44100,
    max_lag_seconds: int = 0.02,
    window_size_seconds: float = 0.02,
    stride_seconds: float = 0.01,
    epsilon: float = 1e-10,
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

    cross_correlogram = np.array(
        [
            (
                np.correlate(
                    xnk := x[start : start + window_size_samples],
                    ynkt := y[start : start + window_size_samples],
                    mode='full',
                )
                / (
                    np.sqrt(np.dot(xnk, xnk) * np.dot(ynkt, ynkt))
                    + epsilon  # Prevent division by zero
                )
            )[:num_lags]
            for start in window_indexes
        ]
    )

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
