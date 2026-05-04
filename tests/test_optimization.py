import pytest
import scipy.io.wavfile as wavfile

from vndecorrelate.decorrelation import HaasEffect, VelvetNoise
from vndecorrelate.optimization import optimize_haas_delay, optimize_velvet_noise
from vndecorrelate.utils.dsp import mono_to_stereo
from vndecorrelate.utils.plot import plot_polar_sample, plot_signal


@pytest.mark.skip
def test__optimize_velvet_noise(viola_signal):
    fs, viola = viola_signal

    if viola.ndim == 1:
        viola = mono_to_stereo(viola)

    duration_seconds = 0.03
    num_impulses = 15

    kappa = optimize_velvet_noise(
        viola,
        fs,
        duration_seconds=duration_seconds,
        num_impulses=num_impulses,
    )
    vnd = VelvetNoise(
        sample_rate_hz=fs,
        duration_seconds=duration_seconds,
        num_impulses=num_impulses,
        log_distribution_strength=kappa,
        filtered_channels=(0,),
        mode='LR',
        seed=1,
    )
    output_signal = vnd.decorrelate(viola)

    print(f'Optimized Kappa: {kappa}')

    wavfile.write('audio/viola_opt.wav', fs, output_signal)

    plot_signal(
        vnd.FIR,
        title='Optimized Velvet Noise Sequence',
    )

    plot_polar_sample(output_signal, title='VN Optimized Vectorscope')


@pytest.mark.skip
def test__optimize_haas_delay(viola_signal):
    fs, viola = viola_signal

    if viola.ndim == 1:
        viola = mono_to_stereo(viola)

    max_delay_seconds = 0.03

    tau = optimize_haas_delay(
        viola,
        fs,
        max_delay_seconds=max_delay_seconds,
    )
    hed = HaasEffect(
        sample_rate_hz=fs,
        delay_time_seconds=tau,
        mode='LR',
    )
    output_signal = hed.decorrelate(viola)

    print(f'Optimized Tau: {tau}')

    wavfile.write('audio/viola_haas.wav', fs, output_signal)

    plot_polar_sample(output_signal, title='HE Optimized Vectorscope')
