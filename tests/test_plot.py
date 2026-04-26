import numpy as np
import pytest
import scipy.io.wavfile as wavfile
from scipy import signal

from tests import IN_GHA
from vndecorrelate.decorrelation import (
    VelvetNoise,
    WhiteNoise,
    generate_velvet_noise,
)
from vndecorrelate.utils.dsp import (
    cross_correlogram,
    generate_decay_envelope,
    sine_sweep,
)
from vndecorrelate.utils.plot import (
    plot_correlogram,
    plot_polar_sample,
    plot_signal,
    plot_spectrogram,
)


@pytest.mark.skipif(IN_GHA, reason='Skipping in CI')
def test_plots_basic():
    # simply run the main function to ensure no errors.
    sample_rate, sig_float32 = wavfile.read('audio/viola.wav')

    segment_envelope = generate_decay_envelope(4, 0.5)

    seed = 2

    plot_signal(
        generate_velvet_noise(
            sample_rate_hz=sample_rate,
            duration_seconds=0.03,
            num_impulses=30,
            num_outs=1,
            log_distribution_strength=0.0,
            segment_envelope=(),
            seed=seed,
        ),
        title='Basic Velvet Noise Sequence',
    )

    plot_signal(
        generate_velvet_noise(
            sample_rate_hz=sample_rate,
            duration_seconds=0.03,
            num_impulses=30,
            num_outs=1,
            log_distribution_strength=0.0,
            seed=seed,
        ),
        title='Segmented Decaying Velvet Noise Sequence',
    )

    plot_signal(
        generate_velvet_noise(
            sample_rate_hz=sample_rate,
            duration_seconds=0.03,
            num_impulses=30,
            num_outs=1,
            log_distribution_strength=0.5,
            segment_envelope=segment_envelope,
            seed=seed,
        ),
        title='Segmented Decaying Log Distributed Velvet Noise Sequence',
    )

    vnd = VelvetNoise(
        sample_rate_hz=sample_rate,
        duration_seconds=0.03,
        num_impulses=30,
        num_outs=2,
        log_distribution_strength=1.0,
        seed=seed,
    )
    result = vnd.decorrelate(sig_float32)
    vns = vnd.FIR

    plot_signal(vns[:, 0], title='Velvet Noise Sequence L')

    plot_signal(vns[:, 1], title='Velvet Noise Sequence R')

    fir = generate_velvet_noise(
        sample_rate_hz=sample_rate,
        duration_seconds=0.03,
        num_impulses=30,
        num_outs=2,
        log_distribution_strength=0.5,
        segment_envelope=segment_envelope,
        seed=seed,
    )

    plot_signal(fir[:, 0], title='Generated Velvet Noise Sequence L')
    plot_signal(fir[:, 1], title='Generated Velvet Noise Sequence R')

    plot_signal(sig_float32, title='Input')
    plot_signal(result, title='Output')

    assert True


@pytest.mark.skipif(IN_GHA, reason='Skipping in CI')
def test_plot_signal():
    x = np.random.uniform(size=100)
    plot_signal(x)

    assert True


@pytest.mark.skipif(IN_GHA, reason='Skipping in CI')
def test__plot_sine_sweep_correlogram():
    fs = 16000
    _sine_sweep = sine_sweep(
        start_freq_hz=20,
        end_freq_hz=8000,
        duration_seconds=5,
        sample_rate_hz=fs,
    )
    plot_spectrogram(
        signal.spectrogram(_sine_sweep, fs=fs)[-1], title='Sine Sweep Signal'
    )
    correlogram = cross_correlogram(
        _sine_sweep,
        _sine_sweep,
        sample_rate_hz=fs,
        max_lag_seconds=0.02,
        window_size_seconds=0.02,
        stride_seconds=0.01,
    )
    plot_correlogram(
        correlogram,
        lag_seconds=0.02,
        time_seconds=5,
        title='Sine-sweep Auto Correlogram',
    )
    assert True


@pytest.mark.skipif(IN_GHA, reason='Skipping in CI')
def test__plot_correlogram_from_file():
    fs, input_signal = wavfile.read('audio/viola.wav')

    duration_seconds = 0.03

    vnd = VelvetNoise(
        sample_rate_hz=fs,
        duration_seconds=duration_seconds,
        num_impulses=30,
        seed=1,
        log_distribution_strength=1.0,
    )

    vnd_output = vnd(input_signal)
    correlogram = cross_correlogram(
        vnd_output[:, 0],
        vnd_output[:, 1],
        sample_rate_hz=fs,
        max_lag_seconds=0.02,
        window_size_seconds=0.02,
        stride_seconds=0.01,
    )
    plot_correlogram(
        correlogram,
        lag_seconds=0.02,
        time_seconds=5,
        title='Viola Velvet Noise Cross Correlogram',
    )

    wnd = WhiteNoise(
        sample_rate_hz=fs,
        duration_seconds=duration_seconds,
        seed=1,
    )

    wnd_output = wnd(input_signal)
    correlogram = cross_correlogram(
        wnd_output[:, 0],
        wnd_output[:, 1],
        sample_rate_hz=fs,
        max_lag_seconds=0.02,
        window_size_seconds=0.02,
        stride_seconds=0.01,
    )
    plot_correlogram(
        correlogram,
        lag_seconds=0.02,
        time_seconds=5,
        title='Viola White Noise Cross Correlogram',
    )

    plot_polar_sample(vnd_output)

    assert True
