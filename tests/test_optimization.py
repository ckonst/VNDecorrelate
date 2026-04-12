import pytest
import scipy.io.wavfile as wavfile

from vndecorrelate.decorrelation import VelvetNoise
from vndecorrelate.optimization import optimize_velvet_noise
from vndecorrelate.utils.plot import plot_polar_sample, plot_signal


@pytest.mark.skip
def test__optimize_velvet_noise():
    sample_rate_hz, input_signal = wavfile.read('audio/guitar.wav')

    duration_seconds = 0.03
    num_impulses = 30

    kappa = optimize_velvet_noise(
        input_signal,
        sample_rate_hz,
        duration_seconds=duration_seconds,
        num_impulses=num_impulses,
    )
    vnd = VelvetNoise(
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
        num_impulses=num_impulses,
        log_distribution_strength=kappa,
        seed=1,
    )
    output_signal = vnd.decorrelate(input_signal)

    print(f'Optimized Kappa: {kappa}')

    wavfile.write('audio/guitar_opt.wav', sample_rate_hz, output_signal)

    plot_signal(
        vnd.FIR,
        title='Optimized Velvet Noise Sequence',
    )

    plot_polar_sample(output_signal, title='VN Optimized Vectorscope')
