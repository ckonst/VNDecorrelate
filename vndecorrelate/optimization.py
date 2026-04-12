import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from vndecorrelate.decorrelation import Decorrelator, VelvetNoise
from vndecorrelate.utils.dsp import EPSILON, polar_coordinates


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

    thetas, radii, weights = polar_coordinates(output_signal[:, 0], output_signal[:, 1])

    # Even moment: variance => maximize spread
    spread = float(np.sum(weights * thetas**2))

    # 1st odd moment: weighted mean => optimize to zero
    mean_theta = float(np.sum(weights * thetas))

    # 3rd odd moment: weighted skewness => optimize to zero
    # normalize by the proportional amount of spread in the distribution
    skewness = float(np.sum(weights * thetas**3)) / (max(spread, EPSILON) ** 1.5)

    # Constraint: max angular exceedance
    violation = float(np.max(np.abs(thetas)) - angle_limit)
    constraint_penalty = lambda_penalty * max(0.0, violation) ** 2

    objective = (
        spread
        - lambda_mean * mean_theta**2
        - lambda_skew * skewness**2
        - constraint_penalty
    )

    return -objective  # minimizer convention


def optimize_velvet_noise(
    input_signal: NDArray,
    sample_rate_hz: int,
    duration_seconds: float,
    num_impulses: int,
    seed: int = 1,
    grid_size: int = 400,
    angle_limit: float = np.pi / 4,
    lambda_mean: float = 5.0,
    lambda_skew: float = 2.0,
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
            seed=seed,
        )
        for kappa in kappas
    ]

    # --- Grid scan ---
    scores = np.array(
        [
            symmetry_aware_objective(
                input_signal,
                velvet_noise,
                angle_limit=angle_limit,
                lambda_mean=lambda_mean,
                lambda_skew=lambda_skew,
            )
            for velvet_noise in velvet_noise_decorrelators
        ]
    )

    # Identify local minima (best candidate regions)
    # A local min at index i: scores[i] < scores[i-1] and scores[i] < scores[i+1]
    local_minima = [
        i
        for i in range(1, grid_size - 1)
        if scores[i] < scores[i - 1] and scores[i] < scores[i + 1]
    ]
    if not local_minima:
        local_minima = [int(np.argmin(scores))]

    best_kappa, best_score = 0.0, np.inf

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
                    seed=seed,
                ),
                angle_limit=angle_limit,
                lambda_mean=lambda_mean,
                lambda_skew=lambda_skew,
            ),
            bounds=(low, high),
            method='bounded',
            options={'xatol': 1e-4},
        )

        if result.fun < best_score:
            best_score = result.fun
            best_kappa = result.x

    return best_kappa
