import pytest
import scipy.io.wavfile as wavfile
import numpy as np
from matplotlib import pyplot as plt

from vndecorrelate.decorrelation import HaasEffect, VelvetNoise
from vndecorrelate import optimization
from vndecorrelate.optimization import optimize_haas_delay, optimize_velvet_noise
from vndecorrelate.utils.dsp import mono_to_stereo
from vndecorrelate.utils.plot import (
    plot_lissajous,
    plot_polar_level,
    plot_polar_sample,
    plot_signal,
)


def test__optimize_haas_delay_uses_configured_delay_bounds(monkeypatch):
    min_delay_seconds = 0.01
    max_delay_seconds = 0.03
    grid_size = 3

    monkeypatch.setattr(
        optimization,
        'grid_scan',
        lambda *_args, **_kwargs: np.zeros(grid_size),
    )

    def fake_optimize_local_minima(local_minima, scalars, received_grid_size, _):
        assert local_minima == [0]
        assert received_grid_size == grid_size
        np.testing.assert_allclose(
            scalars,
            [min_delay_seconds, 0.02, max_delay_seconds],
        )
        return scalars[1]

    monkeypatch.setattr(optimization, 'optimize_local_minima', fake_optimize_local_minima)

    result = optimize_haas_delay(
        input_signal=np.ones((8, 2)),
        sample_rate_hz=100,
        min_delay_seconds=min_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        grid_size=grid_size,
    )

    assert min_delay_seconds <= result <= max_delay_seconds


@pytest.mark.parametrize(
    ('min_delay_seconds', 'max_delay_seconds', 'grid_size', 'message'),
    [
        (-0.01, 0.03, 3, 'min_delay_seconds'),
        (np.nan, 0.03, 3, 'min_delay_seconds'),
        (0.03, 0.03, 3, 'max_delay_seconds'),
        (0.04, 0.03, 3, 'max_delay_seconds'),
        (0.01, np.inf, 3, 'max_delay_seconds'),
        (0.01, 0.03, 1, 'grid_size'),
        (0.01, 0.03, 2.5, 'grid_size'),
    ],
)
def test__optimize_haas_delay_rejects_invalid_search_parameters(
    min_delay_seconds, max_delay_seconds, grid_size, message
):
    with pytest.raises(ValueError, match=message):
        optimize_haas_delay(
            input_signal=np.ones((8, 2)),
            sample_rate_hz=100,
            min_delay_seconds=min_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            grid_size=grid_size,
        )


@pytest.mark.skip
def test__optimize_velvet_noise(viola_signal):
    fs, viola = viola_signal

    if viola.ndim == 1:
        viola = mono_to_stereo(viola)

    duration_seconds = 0.03
    num_impulses = 15

    kappa = optimize_velvet_noise(
        input_signal=viola,
        sample_rate_hz=fs,
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

    wavfile.write('audio/viola_optimized_vnd.wav', fs, output_signal)

    plot_signal(
        vnd.FIR,
        title='Optimized Velvet Noise Sequence',
    )
    plt.savefig('./tests/plots/Optimized Velvet Noise Sequence.png')
    plt.close()

    unoptimized_output_signal = VelvetNoise(
        sample_rate_hz=fs,
        duration_seconds=duration_seconds,
        num_impulses=30,
        log_distribution_strength=1.0,
        filtered_channels=(0,),
        mode='LR',
        seed=1,
    )(viola)

    background_color = '#0a0a0f'
    _, all_axes = plt.subplots(1, 2, figsize=(22, 6), facecolor=background_color)

    plot_polar_sample(
        unoptimized_output_signal,
        title='Unoptimized VN Polar Sample',
        axes=all_axes[0],
        mode='both',
    )
    plot_polar_sample(
        output_signal,
        title='VN Optimized Polar Sample',
        axes=all_axes[1],
        mode='both',
    )
    plt.savefig('./tests/plots/VN Optimized Polar Sample.png')
    plt.close()

    _, all_axes = plt.subplots(1, 2, figsize=(22, 6), facecolor=background_color)

    plot_polar_level(
        unoptimized_output_signal,
        title='Unoptimized VN Polar Level',
        axes=all_axes[0],
    )
    plot_polar_level(
        output_signal,
        title='VN Optimized Polar Level',
        axes=all_axes[1],
    )
    plt.savefig('./tests/plots/VN Optimized Polar Level.png')
    plt.close()

    _, all_axes = plt.subplots(1, 2, figsize=(22, 6), facecolor=background_color)

    plot_lissajous(
        unoptimized_output_signal,
        title='Unoptimized VN Lissajous',
        axes=all_axes[0],
    )
    plot_lissajous(
        output_signal,
        title='VN Optimized Lissajous',
        axes=all_axes[1],
    )
    plt.savefig('./tests/plots/VN Optimized Lissajous.png')
    plt.close()


@pytest.mark.skip
def test__optimize_haas_delay(viola_signal):
    fs, viola = viola_signal

    if viola.ndim == 1:
        viola = mono_to_stereo(viola)

    min_delay_seconds = 0.01
    max_delay_seconds = 0.03

    tau = optimize_haas_delay(
        input_signal=viola,
        sample_rate_hz=fs,
        max_delay_seconds=max_delay_seconds,
        min_delay_seconds=min_delay_seconds,
    )
    hed = HaasEffect(
        sample_rate_hz=fs,
        delay_time_seconds=tau,
        mode='LR',
    )
    output_signal = hed.decorrelate(viola)

    print(f'Optimized Tau: {tau}')

    wavfile.write('audio/viola_optimized_hed.wav', fs, output_signal)

    plot_polar_sample(output_signal, title='HE Optimized Vectorscope', mode='scatter')
    plt.savefig('./tests/plots/HE Optimized Vectorscope.png')
    plt.close()
