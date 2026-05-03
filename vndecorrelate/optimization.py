from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from vndecorrelate.decorrelation import Decorrelator, HaasEffect, VelvetNoise
from vndecorrelate.utils.dsp import EPSILON, polar_coordinates


def left_right_correlation(stereo_signal: NDArray) -> float:
    """Return the dot product between the normalized left and right channels: [-1.0, 1.0]"""
    return np.dot(
        stereo_signal[:, 0] / (np.linalg.norm(stereo_signal[:, 0]) + EPSILON),
        stereo_signal[:, 1] / (np.linalg.norm(stereo_signal[:, 0]) + EPSILON),
    )


def angular_variance(thetas: NDArray, weights: NDArray) -> float:
    """Return the amplitude-weighted angular variance (spread) of the polar samples: E_w[θ²]"""
    return float(np.sum(weights * thetas**2))


def centroid(thetas: NDArray, weights: NDArray) -> float:
    """Return the mean theta (centroid) of the polar samples: E_w[θ]"""
    return float(np.sum(weights * thetas))


def polar_skewness(
    thetas: NDArray, weights: NDArray, angular_variance: NDArray
) -> float:
    """Return the skewness of the polar samples: ``E_w[θ³]``"""

    return (
        float(np.sum(weights * thetas**3))
        /
        # standardize by dividing by σ³ == (σ²)³ᐟ²
        (max(angular_variance, EPSILON) ** 1.5)
    )


def max_angular_exceedance(thetas: NDArray, angle_limit: float) -> float:
    """return the maximum angular exceedance of the polar samples."""
    return max(0.0, float(np.max(np.abs(thetas)) - angle_limit))


def left_right_correlation_objective(
    input_signal: NDArray,
    decorrelator: Decorrelator,
) -> float:
    """Scalar objective for left-right correlation r optimization.

    Returns a value to minimize.
    """
    output_signal = decorrelator.decorrelate(input_signal)

    # left-right correlation: square of dot product => optimize to zero
    dot_product_squared = left_right_correlation(output_signal) ** 2

    return dot_product_squared


def symmetry_aware_objective(
    input_signal: NDArray,
    decorrelator: Decorrelator,
    angle_limit: float = np.pi / 4,
    lambda_mean: float = 5.0,  # centroid penalty weight
    lambda_skew: float = 2.0,  # skewness penalty weight
    lambda_correlation: float = 15.0,  # correlation penalty weight
    lambda_penalty: float = 1e3,  # constrain violation penalty
) -> float:
    """
    Scalar objective function for maximizing stereo width through a Convex Hull area proxy (weighted angular variance).
    Penalizes off-centeredness, skewness, correlation, and angular exceedance of the polar samples.

    Returns a value to minimize (negate of the multi-objective).

    Terms
    -----
    ``+E_w[θ²]``         spread — maximize hull area proxy
    ``−λ₁·(E_w[θ])²``    centroid — keep hull centered on ``θ=0``
    ``−λ₂·(E_w[θ³])²``   skewness — keep left/right halves balanced
    ``-λ₃∙r``            correlation — keep left/right correlation = 0
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


def grid_scan(input_signal: NDArray, decorrelators: list[Decorrelator]) -> NDArray:
    print('Starting Grid Scan')
    return np.array(
        [
            symmetry_aware_objective(input_signal, decorrelator)
            for decorrelator in decorrelators
        ]
    )


def get_local_minima(scores: NDArray, grid_size: int) -> list[int]:
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
    scalars: NDArray[np.floating[Any]],
    grid_size: int,
    scalar_objective: Callable[[float], float],
) -> np.floating[Any]:
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
    input_signal: NDArray,
    sample_rate_hz: int,
    max_delay_seconds: int,
    grid_size: int = 400,
) -> float:
    # tau: length of delay for channel
    # TODO: compute grid_size based on nyquist criterion applied to tau-domain landscape
    taus = np.linspace(0.0, max_delay_seconds, grid_size)

    haas_effect_decorrelators = [
        HaasEffect(
            sample_rate_hz=sample_rate_hz,
            delay_time_seconds=tau,
            mode='LR',
        )
        for tau in taus
    ]

    scores = grid_scan(input_signal, haas_effect_decorrelators)

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
        ),
    )


def optimize_velvet_noise(
    input_signal: NDArray,
    sample_rate_hz: int,
    duration_seconds: float,
    num_impulses: int,
    seed: int = 1,
    grid_size: int = 400,
) -> float:
    # kappa: concentration of impulses toward start of sequence
    # TODO: compute grid_size based on nyquist criterion applied to kappa-domain landscape
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

    scores = grid_scan(input_signal, velvet_noise_decorrelators)

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
        ),
    )
