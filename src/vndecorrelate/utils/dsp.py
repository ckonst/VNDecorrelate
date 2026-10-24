"""Digital-signal-processing helpers for stereo audio analysis."""

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull

EPSILON: float = 1e-10

# An envelope with a single segment to leave a signal unchanged when applied.
IDENTITY_ENVELOPE: tuple[float] = (1.0,)
# An approximation of an exponential decay envelope using 4 equal-length segments, according to Alary et al.
SEGMENTED_DECAY_ENVELOPE: tuple[float] = (0.85, 0.55, 0.35, 0.2)


class NormalizeMode(StrEnum):
    """Channel-layout modes used by normalization functions."""

    STEREO = 'stereo'
    DUAL_MONO = 'dual_mono'


class LayoutMode(StrEnum):
    """Represent stereo channels as left-right or mid-side pairs."""

    LR = 'LR'  # Left-Right
    MS = 'MS'  # Mid-Side


def apply_stereo_width(input_signal: NDArray, width: float) -> None:
    """Adjust a stereo signal's mid-side balance in place.

    ``input_signal`` MUST be a Left-Right stereo signal of shape (n, 2).

    Parameters
    ----------
    input_signal : NDArray
        The original signal.
    width : float
        The percentage to scale the side channel by [0.0, 1.0].

    Returns
    -------
    None
        ``input_signal`` is modified in-place.
    """
    LR_to_MS(input_signal)
    input_signal[:, 0] *= 1.0 - width
    input_signal[:, 1] *= width
    MS_to_LR(input_signal)


def encode_signal_to_side_channel(
    input_signal: NDArray, decorrelated_signal: NDArray
) -> None:
    """Encode ``decorrelated_signal`` as the side channel of ``input_signal``.

    ``input_signal`` and decorrelated_signal MUST be a Left-Right stereo signal of shape (n, 2).
    Assumes that ``input_signal`` is a mono signal duplicated to stereo,
    both channels of ``decorrelated_signal`` will be overwritten.

    Parameters
    ----------
    input_signal : NDArray
        The original signal.
    decorrelated_signal : NDArray
        The decorrelated signal.

    Returns
    -------
    None
        ``decorrelated_signal`` is modified in-place.
    """
    check_stereo(input_signal)
    check_stereo(decorrelated_signal)
    M = np.sum(input_signal, axis=1)
    # diff does col 1 - col 0, so swap columns to get L - R
    S = np.diff(decorrelated_signal[:, [1, 0]], axis=1).squeeze() * 0.5
    decorrelated_signal[:, 0] = (M + S) * 0.5
    decorrelated_signal[:, 1] = (M - S) * 0.5


def to_float32(input_signal: NDArray) -> NDArray[np.float32]:
    """Convert an array to ``float32`` without an unnecessary copy.

    Parameters
    ----------
    input_signal : NDArray
        Array to convert.

    Returns
    -------
    NDArray[np.float32]
        ``input_signal`` represented with the ``float32`` dtype.
    """
    return input_signal.astype(np.float32, copy=False)


def peak_normalize(
    input_signal: NDArray,
    mode: NormalizeMode = NormalizeMode.DUAL_MONO,
    epsilon: float = EPSILON,
) -> None:
    """Peak-normalize an audio signal in place.

    Parameters
    ----------
    input_signal : NDArray
        Mono or stereo audio signal to normalize.
    mode : NormalizeMode
        Whether stereo signals are normalized jointly or per channel.
    epsilon : float
        Small positive value that prevents division by zero.

    Returns
    -------
    None
        ``input_signal`` is modified in-place.
    """
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
    """Match an output signal's RMS level to an input signal in-place.

    Parameters
    ----------
    input_signal : NDArray
        Reference mono or stereo audio signal.
    output_signal : NDArray
        Signal whose RMS level is adjusted.
    mode : NormalizeMode
        Whether stereo signals are normalized jointly or per channel.
    epsilon : float
        Small positive value that prevents division by zero.

    Returns
    -------
    None
        ``output_signal`` is modified in-place.
    """
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
    """Duplicate a mono signal into left and right channels.

    Parameters
    ----------
    input_signal : NDArray
        Mono signal with shape ``(num_samples,)``.

    Returns
    -------
    NDArray
        Stereo signal with shape ``(num_samples, 2)``.
    """
    check_mono(input_signal)
    return np.column_stack((input_signal, input_signal))


def stereo_to_mono(input_signal: NDArray) -> NDArray:
    """Average a stereo signal's channels into a mono signal.

    Parameters
    ----------
    input_signal : NDArray
        Stereo signal with shape ``(num_samples, 2)``.

    Returns
    -------
    NDArray
        Mono signal with shape ``(num_samples,)``.
    """
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

    Returns
    -------
    None
        ``input_signal`` is modified in place.
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

    Returns
    -------
    None
        ``input_signal`` is modified in place.
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
        a strong (exponential) increase.
    size : int
        Number of intervals; the returned array has length ``size + 1``.

    Returns
    -------
    NDArray
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
        # to generate a flat, all 1-valued array, while still parameterized by strength.
        (1.0 + (strength * 99.0))
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
    randoms : NDArray
        Uniform random values in ``[0, 1]``; shape must match
        ``log_distribution`` and ``log_impulse_intervals``.
    log_distribution : NDArray
        Precomputed distribution weights from
        :func:``generate_log_distribution`` describing how density varies
        across the filter.
    log_impulse_intervals : NDArray
        Cumulative impulse interval positions corresponding to the
        ``log_distribution`` values.
    jitter : float
        Scale applied to ``log_distribution`` that controls how much the
        random offsets can perturb each impulse position. A value of ``0.0`` means no jitter.
        Typical usage is to set this to the average impulse interval (samples per impulse).

    Returns
    -------
    NDArray[np.int32]
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
    randoms : NDArray
        Uniform random values in the interval ``[0, 1]``, with the same shape
        as ``impulse_indexes``.
    impulse_indexes : NDArray
        Precomputed impulse indices (for example ``np.arange(n_impulses)``),
        where the first impulse is at index ``0``.
    impulse_interval : float
        Average interval between impulses (filter length divided by the
        number of impulses).

    Returns
    -------
    positions : NDArray[np.int32]
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
    max_lag_seconds: float = 0.02,
    window_size_seconds: float = 0.02,
    stride_seconds: float = 0.01,
    epsilon: float = EPSILON,
) -> NDArray:
    """Compute a windowed, normalized cross-correlogram.

    Parameters
    ----------
    x, y : NDArray
        Equal-length mono signals.
    sample_rate_hz : int
        Signal sample rate in hertz.
    max_lag_seconds : float
        Largest absolute lag to include, in seconds.
    window_size_seconds : float
        Analysis-window duration in seconds.
    stride_seconds : float
        Step between adjacent analysis windows in seconds.
    epsilon : float
        Small positive value that prevents division by zero.

    Returns
    -------
    NDArray
        Normalized correlation values with shape ``(num_windows, num_lags)``.
    """
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
    """Generate an exponential sine sweep.

    Parameters
    ----------
    start_freq_hz : float
        Initial frequency in hertz.
    end_freq_hz : float
        Final frequency in hertz.
    duration_seconds : float
        Sweep duration in seconds.
    sample_rate_hz : int
        Output sample rate in hertz.

    Returns
    -------
    NDArray
        ``float32`` mono signal containing the sweep.
    """
    t = np.linspace(
        0, duration_seconds, int(sample_rate_hz * duration_seconds), endpoint=False
    )
    k = np.log(end_freq_hz / start_freq_hz) / duration_seconds
    sine_sweep = np.sin(2 * np.pi * start_freq_hz * (np.exp(k * t) - 1) / k)
    return sine_sweep.astype(np.float32)


def polar_coordinates(
    left: NDArray,
    right: NDArray,
    mode: LayoutMode = LayoutMode.MS,
    semicircular: bool = True,
    normalize: bool = True,
    compute_weights: bool = True,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray]:
    """Return each sample of the ``left`` and ``right`` channels as polar coordinates,
    by default amplitude weights are also returned: ``(radii, thetas, weights)``.

    Parameters
    ----------
    left : NDArray
        The left channel with values in [-1, 1]
    right : NDArray
        The right channel with values in [-1, 1]
    mode : LayoutMode
        The channel layout to use for calculating angle. Defaults to LayoutMode.MS (Mid-Side).
    semicircular : bool
        Whether to fold the full (-180, 180] range onto [-90, 90]. Default is True.
    normalize : bool
        Whether to normalize the radii component of the polar coordinates. Default is True.
    compute_weights : bool
        Whether to compute and return amplitude weights for the polar coordinates.

    Returns
    -------
    tuple[NDArray, NDArray] or tuple[NDArray, NDArray, NDArray]
        Radii and angles in radians; normalized amplitude weights are included
        when ``compute_weights`` is ``True``.
    """
    # in radians, (-π, +π]
    thetas = (
        np.arctan2(left - right, left + right)
        if mode == LayoutMode.MS
        else np.arctan2(left, right)
    )

    # Fold the full (-180, 180] range onto [-90, 90] by wrapping the
    # bottom half of the circle (inverted-mono region) onto the top half.
    if semicircular:
        thetas = np.where(
            thetas < -np.pi / 2,
            thetas + np.pi,
            np.where(thetas > np.pi / 2, thetas - np.pi, thetas),
        )

    radii = np.sqrt(left**2 + right**2)
    if normalize:
        radii /= radii.max() + EPSILON

    if compute_weights:
        weights = radii / (radii.sum() + EPSILON)  # amplitude-weights

        return radii, thetas, weights
    return radii, thetas


def polar_to_cartesian(
    angles_degrees: NDArray, radii: NDArray
) -> tuple[NDArray, NDArray]:
    """Convert vectorscope polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    angles_degrees : NDArray
        Vectorscope angles in degrees, where zero points upward.
    radii : NDArray
        Radius for each angle.

    Returns
    -------
    tuple[NDArray, NDArray]
        Cartesian ``x`` and ``y`` coordinates.
    """
    thetas = np.radians(angles_degrees)
    return radii * np.sin(thetas), radii * np.cos(thetas)


def exponential_decay(t: float, k: float = 2) -> float:
    """Evaluate an exponential-decay curve.

    Parameters
    ----------
    t : float
        Time coordinate.
    k : float
        Positive decay rate.

    Returns
    -------
    float
        ``exp(-k * t)``.
    """
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
    tuple[float, ...]
        A tuple of ``num_segments`` floats that define the decay envelope.
    """
    return tuple(
        exponential_decay((t / num_segments) + (segment_position * 1 / num_segments))
        for t in range(num_segments)
    )


def radians_to_degrees(radians: NDArray) -> NDArray:
    """Convert radians to vectorscope degrees clipped to ``[-90, 90]``.

    Parameters
    ----------
    radians : NDArray
        Angles in radians.

    Returns
    -------
    NDArray
        Corresponding clipped angles in degrees.
    """
    return np.clip(
        np.degrees(radians), -90.0, 90.0
    )  # Clip to ±90° (handles phase-inverted / out-of-phase samples gracefully)


def amplitude_to_db(input_signal: NDArray) -> NDArray:
    """Convert linear amplitudes to decibels.

    Parameters
    ----------
    input_signal : NDArray
        Non-negative linear amplitudes.

    Returns
    -------
    NDArray
        Amplitudes expressed as ``20 * log10(amplitude + EPSILON)``.
    """
    return 20 * np.log10(input_signal + EPSILON)


def power_to_db(input_signal: NDArray) -> NDArray:
    """Convert linear power values to decibels.

    Parameters
    ----------
    input_signal : NDArray
        Non-negative linear power values.

    Returns
    -------
    NDArray
        Powers expressed as ``10 * log10(power + EPSILON)``.
    """
    return 10 * np.log10(input_signal + EPSILON)


def generate_convex_hull(input_signal: NDArray) -> NDArray:
    """Compute ordered convex-hull vertices for Cartesian points.

    Parameters
    ----------
    input_signal : NDArray
        Cartesian points with shape ``(num_points, 2)``.

    Returns
    -------
    NDArray
        Non-origin hull vertices sorted by their polar angle.
    """
    convex_hull = ConvexHull(input_signal)
    hull_vertices = input_signal[convex_hull.vertices]

    # Drop the origin vertex -- it was only needed for geometric
    # anchoring. Keeping it in the outline draws a line back through
    # the center of the plot.
    is_not_origin = ~(
        (np.abs(hull_vertices[:, 0]) < EPSILON)
        & (np.abs(hull_vertices[:, 1]) < EPSILON)
    )
    hull_vertices = hull_vertices[is_not_origin]

    # hull vertices are guarenteed
    # Sort by angle so the polygon outline draws correctly.
    vertex_angles = np.arctan2(hull_vertices[:, 1], hull_vertices[:, 0])
    sorted_order = np.argsort(vertex_angles)
    return hull_vertices[sorted_order]


def generate_pseudo_hull(
    sample_angles_deg: NDArray,
    sample_radii: NDArray,
    num_bins: int = 360,
    percentile: float = 95.0,
    smoothing_sigma: float = 4.0,
    use_convex_hull: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Build a polar envelope from sample polar coordinates.


    Steps:
      1. Bin the 180-degree angular range into num_bins slices.
      2. Take the given percentile of radii within each bin.
      3. Apply Gaussian smoothing around the ring.
      4. Optionally replace the smooth envelope with its geometric convex hull.

    Parameters
    ----------
    sample_angles_deg : NDArray
        Per-sample azimuth angles in degrees, range [-90, +90].
    sample_radii : NDArray
        Per-sample normalized radii, range [0, 1].
    num_bins : int
        Number of angular bins across the 180-degree range.
    percentile : float
        Radial percentile used as the envelope ceiling (0-100).
        95 gives a near-peak envelope; 50 gives the median shape.
    smoothing_sigma : float
        Standard deviation (in bin units) of the Gaussian smoothing kernel.
    use_convex_hull : bool
        If True, compute the geometric convex hull of the percentile envelope
        points and use that outline. If False, use the smoothed envelope directly.

    Returns
    -------
    hull_angles_deg : NDArray
        Angles of the hull vertices, closed (first == last).
    hull_radii : NDArray
        Radii of the hull vertices, closed (first == last).
    bin_center_angles : NDArray
        Center angle of every bin (useful for the inner RMS overlay).
    bin_percentile_radii : NDArray
        Percentile radius for every bin.
    """
    bin_edges = np.linspace(-90.0, 90.0, num_bins + 1)
    bin_center_angles = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # generate a qth percentile envelope from each bin.
    bin_percentile_radii = np.zeros(num_bins)
    for bin_index in range(num_bins):
        mask = (sample_angles_deg >= bin_edges[bin_index]) & (
            sample_angles_deg < bin_edges[bin_index + 1]
        )
        if mask.any():
            bin_percentile_radii[bin_index] = np.percentile(
                sample_radii[mask], percentile
            )

    if smoothing_sigma > 0:
        bin_percentile_radii = gaussian_filter1d(
            bin_percentile_radii, sigma=smoothing_sigma, mode='wrap'
        )

    if not use_convex_hull or np.count_nonzero(bin_percentile_radii) < 3:
        hull_angles_deg = np.append(bin_center_angles, bin_center_angles[0])
        hull_radii = np.append(bin_percentile_radii, bin_percentile_radii[0])
        return hull_angles_deg, hull_radii, bin_center_angles, bin_percentile_radii

    # Convert binned envelope to Cartesian for convex hull computation.
    # Use the intermediate (pre-rotation) angle space where bins are
    # uniformly spaced from -90 to +90 degrees of standard math angle.
    envelope_x = bin_percentile_radii * np.cos(np.radians(bin_center_angles))
    envelope_y = bin_percentile_radii * np.sin(np.radians(bin_center_angles))

    # Include the origin so the hull is always grounded at the base.
    hull_input_points = np.vstack(
        [np.column_stack([envelope_x, envelope_y]), [0.0, 0.0]]
    )

    try:
        hull_vertices = generate_convex_hull(hull_input_points)

        hull_angles_deg = np.degrees(
            np.arctan2(hull_vertices[:, 1], hull_vertices[:, 0])
        )
        hull_radii = np.hypot(hull_vertices[:, 0], hull_vertices[:, 1])

        # Close the loop
        hull_angles_deg = np.append(hull_angles_deg, hull_angles_deg[0])
        hull_radii = np.append(hull_radii, hull_radii[0])

    except Exception:
        # Fall back to the smooth envelope if the hull computation fails.
        hull_angles_deg = np.append(bin_center_angles, bin_center_angles[0])
        hull_radii = np.append(bin_percentile_radii, bin_percentile_radii[0])

    return hull_angles_deg, hull_radii, bin_center_angles, bin_percentile_radii
