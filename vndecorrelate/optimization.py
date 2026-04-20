import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from vndecorrelate.decorrelation import Decorrelator, HaasEffect, VelvetNoise
from vndecorrelate.utils.dsp import EPSILON, polar_coordinates


def left_right_correlation_objective(
    input_signal: NDArray,
    decorrelator: Decorrelator,
) -> float:
    output_signal = decorrelator.decorrelate(input_signal)

    # left-right correlation: square of dot product => optimize to zero
    abs_dot_product = np.dot(output_signal[:, 0], output_signal[:, 1]) ** 2

    return abs_dot_product


def symmetry_aware_objective(
    input_signal: NDArray,
    decorrelator: Decorrelator,
    angle_limit: float = np.pi / 4,
    lambda_mean: float = 5.0,  # centroid penalty weight
    lambda_skew: float = 2.0,  # skewness penalty weight
    lambda_penalty: float = 1e3,  # constrain violation penalty
) -> float:
    """
    Scalar objective for log impulse concentration κ optimization.

    Returns a value to MINIMIZE (negate of the multi-objective).

    Terms
    -----
    +E_w[θ²]         spread — maximize hull area proxy
    −λ₁·(E_w[θ])²    centroid — keep hull centered on θ=0
    −λ₂·(E_w[θ³])²   skewness — keep left/right halves balanced
    -penalty         angle constraint via quadratic penalty
    """

    output_signal = decorrelator.decorrelate(input_signal)

    radii, thetas, weights = polar_coordinates(
        output_signal[:, 0], output_signal[:, 1], normalize=False
    )

    # Even moment: variance => maximize spread
    spread = float(np.sum(weights * thetas**2))

    # 1st odd moment: weighted mean => optimize to zero
    mean_theta_penalty = lambda_mean * float(np.sum(weights * thetas)) ** 2

    # 3rd odd moment: weighted skewness => optimize to zero
    # normalize by the proportional amount of spread in the distribution
    skewness_penalty = (
        lambda_skew
        * (float(np.sum(weights * thetas**3)) / (max(spread, EPSILON) ** 1.5)) ** 2
    )

    # Constraint: max angular exceedance
    violation = float(np.max(np.abs(thetas)) - angle_limit)
    constraint_penalty = lambda_penalty * max(0.0, violation) ** 2

    objective = spread - mean_theta_penalty - skewness_penalty - constraint_penalty

    return -objective  # minimizer convention


def grid_scan(input_signal: NDArray, decorrelators: list[Decorrelator]) -> NDArray:
    return np.array(
        [
            left_right_correlation_objective(input_signal, decorrelator)
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

    print('Starting Grid Scan')
    scores = grid_scan(input_signal, haas_effect_decorrelators)

    # Identify local minima (best candidate regions)
    # A local min at index i: scores[i] < scores[i-1] and scores[i] < scores[i+1]
    local_minima = get_local_minima(scores, grid_size)

    best_tau, best_score = 0.0, np.inf

    print('Starting Local Minima optimization')
    for i in local_minima:
        low = taus[max(0, i - 1)]
        high = taus[min(grid_size - 1, i + 1)]

        result = minimize_scalar(
            lambda tau: symmetry_aware_objective(
                input_signal,
                HaasEffect(
                    sample_rate_hz=sample_rate_hz,
                    delay_time_seconds=tau,
                    mode='LR',
                ),
            ),
            bounds=(low, high),
            method='bounded',
            options={'xatol': 1e-4},
        )

        if result.fun < best_score:
            best_score = result.fun
            best_tau = result.x

    return best_tau


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

    print('Starting Grid Scan')
    scores = grid_scan(input_signal, velvet_noise_decorrelators)

    # Identify local minima (best candidate regions)
    # A local min at index i: scores[i] < scores[i-1] and scores[i] < scores[i+1]
    local_minima = get_local_minima(scores, grid_size)

    best_kappa, best_score = 0.0, np.inf

    print('Starting Local Minima optimization')
    for i in local_minima:
        low = kappas[max(0, i - 1)]
        high = kappas[min(grid_size - 1, i + 1)]

        result = minimize_scalar(
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
            bounds=(low, high),
            method='bounded',
            options={'xatol': 1e-4},
        )

        if result.fun < best_score:
            best_score = result.fun
            best_kappa = result.x

    return best_kappa
