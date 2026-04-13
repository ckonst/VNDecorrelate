import scipy.io.wavfile as wavfile

from vndecorrelate.decorrelation import VelvetNoise
from vndecorrelate.optimization import optimize_velvet_noise
from vndecorrelate.utils.dsp import mono_to_stereo
from vndecorrelate.utils.plot import plot_polar_sample, plot_signal


def test__optimize_velvet_noise():
    sample_rate_hz, input_signal = wavfile.read('audio/pop_shuffle.wav')

    if input_signal.ndim == 1:
        input_signal = mono_to_stereo(input_signal)

    duration_seconds = 0.03
    num_impulses = 15

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
        mode='LR',
        seed=1,
    )
    output_signal = vnd.decorrelate(input_signal)

    print(f'Optimized Kappa: {kappa}')

    wavfile.write('audio/pop_shuffle_opt.wav', sample_rate_hz, output_signal)

    plot_signal(
        vnd.FIR,
        title='Optimized Velvet Noise Sequence',
    )

    plot_polar_sample(output_signal, title='VN Optimized Vectorscope')


test__optimize_velvet_noise()
