"""Objective functions and scalar optimizers for audio decorrelators."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from vndecorrelate.decorrelation import Decorrelator, HaasEffect, VelvetNoise
from vndecorrelate.utils.dsp import EPSILON, polar_coordinates


def left_right_correlation(stereo_signal: NDArray) -> float:
    """Compute the normalized correlation of a stereo signal's channels.

    Parameters
    ----------
    stereo_signal : NDArray
        Stereo signal with shape ``(num_samples, 2)``.

    Returns
    -------
    float
        The left-right correlation coefficient in ``[-1.0, 1.0]``.
    """
    return np.dot(
        stereo_signal[:, 0] / (np.linalg.norm(stereo_signal[:, 0]) + EPSILON),
        stereo_signal[:, 1] / (np.linalg.norm(stereo_signal[:, 0]) + EPSILON),
    )


def angular_variance(thetas: NDArray, weights: NDArray) -> float:
    """Compute the amplitude-weighted second angular moment.

    Parameters
    ----------
    thetas : NDArray
        Polar angles in radians.
    weights : NDArray
        Non-negative weights corresponding to ``thetas``.

    Returns
    -------
    float
        The weighted second moment, ``E_w[theta**2]``.
    """
    return float(np.sum(weights * thetas**2))


def centroid(thetas: NDArray, weights: NDArray) -> float:
    """Compute the amplitude-weighted angular centroid.

    Parameters
    ----------
    thetas : NDArray
        Polar angles in radians.
    weights : NDArray
        Non-negative weights corresponding to ``thetas``.

    Returns
    -------
    float
        The weighted first moment, ``E_w[theta]``.
    """
    return float(np.sum(weights * thetas))


def polar_skewness(thetas: NDArray, weights: NDArray, angular_variance: float) -> float:
    """Compute standardized amplitude-weighted angular skewness.

    Parameters
    ----------
    thetas : NDArray
        Polar angles in radians.
    weights : NDArray
        Non-negative weights corresponding to ``thetas``.
    angular_variance : float
        Weighted second moment of ``thetas``.

    Returns
    -------
    float
        The standardized weighted third moment.
    """
    return (
        float(np.sum(weights * thetas**3))
        /
        # standardize by dividing by σ³ == (σ²)³ᐟ²
        (max(angular_variance, EPSILON) ** 1.5)
    )


def max_angular_exceedance(thetas: NDArray, angle_limit: float) -> float:
    """Compute how far the largest polar angle exceeds a limit.

    Parameters
    ----------
    thetas : NDArray
        Polar angles in radians.
    angle_limit : float
        Largest permitted absolute angle in radians.

    Returns
    -------
    float
        Zero when every angle is within the limit; otherwise the maximum
        exceedance in radians.
    """
    return max(0.0, float(np.max(np.abs(thetas)) - angle_limit))


def symmetry_aware_objective(
    input_signal: NDArray,
    decorrelator: Decorrelator,
    *,
    angle_limit: float,
    lambda_mean: float,  # centroid penalty weight
    lambda_skew: float,  # skewness penalty weight
    lambda_correlation: float,  # correlation penalty weight
    lambda_penalty: float,  # constrain violation penalty
) -> float:
    r"""Evaluate a scalar objective for stereo-image optimization.

    The objective rewards weighted angular spread and penalizes an off-center
    image, skewness, left-right correlation, and angles beyond ``angle_limit``.

    Parameters
    ----------
    input_signal : NDArray
        Input audio signal passed to ``decorrelator``.
    decorrelator : Decorrelator
        Decorrelator whose output is evaluated.
    angle_limit : float
        Largest permitted absolute polar angle in radians.
    lambda_mean : float
        Weight for the angular-centroid penalty.
    lambda_skew : float
        Weight for the angular-skewness penalty.
    lambda_correlation : float
        Weight for the left-right correlation penalty.
    lambda_penalty : float
        Weight for the angular-limit violation penalty.

    Returns
    -------
    float
        Negated objective value suitable for minimization.

    Terms
    -----
    ``+E_w[θ²]``         spread — maximize hull area proxy\n
    ``−λ₁·(E_w[θ])²``    centroid — keep hull centered on ``θ = 0``\n
    ``−λ₂·(E_w[θ³])²``   skewness — keep left/right halves balanced\n
    ``-λ₃∙r²``            correlation — minimize squared left/right correlation ``r² → 0``\n
    ``-penalty``         angle constraint via quadratic penalty
    """
    output_signal = decorrelator.decorrelate(input_signal)

    radii, thetas, weights = polar_coordinates(
        output_signal[:, 0], output_signal[:, 1], normalize=False
    )

    # Even moment: variance => maximize spread
    spread = angular_variance(thetas, weights)

    # 1st odd moment: weighted mean => optimize to zero
    mean_theta_penalty = lambda_mean * centroid(thetas, weights) ** 2

    # 3rd odd moment: weighted skewness => optimize to zero
    # normalize by the proportional amount of spread in the distribution
    skewness_penalty = lambda_skew * polar_skewness(thetas, weights, spread) ** 2

    # left-right correlation: dot product of L and R => optimize to zero
    lr_correlation_penalty = (
        lambda_correlation * left_right_correlation(output_signal) ** 2
    )

    # Constraint: max angular exceedance
    constraint_penalty = (
        lambda_penalty * max_angular_exceedance(thetas, angle_limit) ** 2
    )

    objective = (
        spread
        - mean_theta_penalty
        - skewness_penalty
        - lr_correlation_penalty
        - constraint_penalty
    )

    return -objective  # minimizer convention


def grid_scan(
    input_signal: NDArray, decorrelators: list[Decorrelator], **kwargs: object
) -> NDArray:
    """Evaluate an objective over a collection of decorrelators.

    Parameters
    ----------
    input_signal : NDArray
        Input audio signal supplied to every decorrelator.
    decorrelators : list[Decorrelator]
        Candidate decorrelators to evaluate.
    **kwargs
        Keyword arguments forwarded to :func:`symmetry_aware_objective`.

    Returns
    -------
    NDArray
        One objective value per candidate decorrelator.
    """
    print('Starting Grid Scan')
    return np.array(
        [
            symmetry_aware_objective(input_signal, decorrelator, **kwargs)
            for decorrelator in decorrelators
        ]
    )


def get_local_minima(scores: NDArray, grid_size: int) -> list[int]:
    """Find local-minimum indexes in a one-dimensional score grid.

    Parameters
    ----------
    scores : NDArray
        Objective values sampled on the grid.
    grid_size : int
        Number of points in ``scores`` to examine.

    Returns
    -------
    list[int]
        Indexes of strict local minima, or the global-minimum index if none
        exist.
    """
    local_minima = [
        i
        for i in range(1, grid_size - 1)
        if scores[i] < scores[i - 1] and scores[i] < scores[i + 1]
    ]
    if not local_minima:
        return [int(np.argmin(scores))]
    return local_minima


def optimize_local_minima(
    local_minima: list[int],
    scalars: NDArray,
    grid_size: int,
    scalar_objective: Callable[[float], float],
) -> np.floating[Any]:
    """Refine the best candidates from a scalar grid search.

    Parameters
    ----------
    local_minima : list[int]
        Grid indexes that bound candidate minima.
    scalars : NDArray
        Scalar values sampled by the initial grid search.
    grid_size : int
        Number of scalar grid points.
    scalar_objective : Callable[[float], float]
        Objective function to minimize.

    Returns
    -------
    numpy.floating
        The scalar value with the lowest refined objective.
    """
    best_scalar, best_score = 0.0, np.inf

    print('Starting Local Minima optimization')
    for i in local_minima:
        low = scalars[max(0, i - 1)]
        high = scalars[min(grid_size - 1, i + 1)]

        result = minimize_scalar(
            fun=scalar_objective,
            bounds=(low, high),
            method='bounded',
            options={'xatol': 1e-4},
        )

        if result.fun < best_score:
            best_score = result.fun
            best_scalar = result.x

    return best_scalar


def optimize_haas_delay(
    *,
    input_signal: NDArray,
    sample_rate_hz: int,
    max_delay_seconds: float,
    grid_size: int = 400,
    angle_limit: float = np.pi / 4,
    lambda_mean: float = 5.0,
    lambda_skew: float = 2.0,
    lambda_correlation: float = 15.0,
    lambda_penalty: float = 1e3,
) -> float:
    r"""Optimize the delay time for a :class:`HaasEffect`.

    Parameters
    ----------
    input_signal : NDArray
        Audio signal to decorrelate and evaluate.
    sample_rate_hz : int
        Input signal sample rate in hertz.
    max_delay_seconds : float
        Upper bound for the delay-time search interval in seconds.
    grid_size : int
        Number of candidates in the initial grid search.
    angle_limit : float
        Largest permitted absolute polar angle in radians.
    lambda_mean, lambda_skew, lambda_correlation, lambda_penalty : float
        Weights for the centroid, skewness, correlation, and angular-limit
        penalties, respectively.

    Terms
    -----
    ``+E_w[θ²]``         spread — maximize hull area proxy\n
    ``−λ₁·(E_w[θ])²``    centroid — keep hull centered on ``θ = 0``\n
    ``−λ₂·(E_w[θ³])²``   skewness — keep left/right halves balanced\n
    ``-λ₃∙r²``            correlation — minimize squared left/right correlation ``r² → 0``\n
    ``-penalty``         angle constraint via quadratic penalty

    Returns
    -------
    float
        Optimized ``delay_time_seconds`` in ``[0.0, max_delay_seconds]``.
    """
    # tau: length of delay for channel
    taus = np.linspace(0.0, max_delay_seconds, grid_size)

    haas_effect_decorrelators = [
        HaasEffect(
            sample_rate_hz=sample_rate_hz,
            delay_time_seconds=tau,
            mode='LR',
        )
        for tau in taus
    ]

    scores = grid_scan(
        input_signal,
        haas_effect_decorrelators,
        angle_limit=angle_limit,
        lambda_mean=lambda_mean,
        lambda_skew=lambda_skew,
        lambda_correlation=lambda_correlation,
        lambda_penalty=lambda_penalty,
    )

    # Identify local minima (best candidate regions)
    # A local min at index i: scores[i] < scores[i-1] and scores[i] < scores[i+1]
    local_minima = get_local_minima(scores, grid_size)

    return optimize_local_minima(
        local_minima,
        taus,
        grid_size,
        lambda tau: symmetry_aware_objective(
            input_signal,
            HaasEffect(
                sample_rate_hz=sample_rate_hz,
                delay_time_seconds=tau,
                mode='LR',
            ),
            angle_limit=angle_limit,
            lambda_mean=lambda_mean,
            lambda_skew=lambda_skew,
            lambda_correlation=lambda_correlation,
            lambda_penalty=lambda_penalty,
        ),
    )


def optimize_velvet_noise(
    *,
    input_signal: NDArray,
    sample_rate_hz: int,
    duration_seconds: float,
    num_impulses: int,
    seed: int = 1,
    grid_size: int = 400,
    angle_limit: float = np.pi / 4,
    lambda_mean: float = 5.0,
    lambda_skew: float = 2.0,
    lambda_correlation: float = 15.0,
    lambda_penalty: float = 1e3,
) -> float:
    r"""Optimize a :class:`VelvetNoise` logarithmic-distribution strength.

    Parameters
    ----------
    input_signal : NDArray
        Audio signal to decorrelate and evaluate.
    sample_rate_hz : int
        Input signal sample rate in hertz.
    duration_seconds : float
        Duration of each candidate velvet-noise FIR in seconds.
    num_impulses : int
        Number of impulses in each candidate FIR.
    seed : int
        Random seed used to construct reproducible candidate filters.
    grid_size : int
        Number of candidates in the initial grid search.
    angle_limit : float
        Largest permitted absolute polar angle in radians.
    lambda_mean, lambda_skew, lambda_correlation, lambda_penalty : float
        Weights for the centroid, skewness, correlation, and angular-limit
        penalties, respectively.

    Terms
    -----
    ``+E_w[θ²]``         spread — maximize hull area proxy\n
    ``−λ₁·(E_w[θ])²``    centroid — keep hull centered on ``θ = 0``\n
    ``−λ₂·(E_w[θ³])²``   skewness — keep left/right halves balanced\n
    ``-λ₃∙r²``            correlation — minimize squared left/right correlation ``r² → 0``\n
    ``-penalty``         angle constraint via quadratic penalty

    Returns
    -------
    float
        Optimized ``log_distribution_strength`` in ``[0.0, 1.0]``.
    """
    kappas = np.linspace(0.0, 1.0, grid_size)

    velvet_noise_decorrelators = [
        VelvetNoise(
            sample_rate_hz=sample_rate_hz,
            duration_seconds=duration_seconds,
            num_impulses=num_impulses,
            log_distribution_strength=kappa,
            normalizer=None,  # speed up optimization by skipping normalization
            filtered_channels=(0,),
            mode='LR',
            seed=seed,
        )
        for kappa in kappas
    ]

    scores = grid_scan(
        input_signal,
        velvet_noise_decorrelators,
        angle_limit=angle_limit,
        lambda_mean=lambda_mean,
        lambda_skew=lambda_skew,
        lambda_correlation=lambda_correlation,
        lambda_penalty=lambda_penalty,
    )

    # Identify local minima (best candidate regions)
    # A local min at index i: scores[i] < scores[i-1] and scores[i] < scores[i+1]
    local_minima = get_local_minima(scores, grid_size)

    return optimize_local_minima(
        local_minima,
        kappas,
        grid_size,
        lambda kappa: symmetry_aware_objective(
            input_signal,
            VelvetNoise(
                sample_rate_hz=sample_rate_hz,
                duration_seconds=duration_seconds,
                num_impulses=num_impulses,
                log_distribution_strength=kappa,
                normalizer=None,  # speed up optimization by skipping normalization
                filtered_channels=(0,),
                mode='LR',
                seed=seed,
            ),
            angle_limit=angle_limit,
            lambda_mean=lambda_mean,
            lambda_skew=lambda_skew,
            lambda_correlation=lambda_correlation,
            lambda_penalty=lambda_penalty,
        ),
    )
