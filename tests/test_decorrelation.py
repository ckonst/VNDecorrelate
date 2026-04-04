import numpy as np
import pytest

from vndecorrelate.decorrelation import (
    HaasEffect,
    SignalChain,
    VelvetNoise,
    WhiteNoise,
    convolve_velvet_noise,
    generate_velvet_noise,
)


def test__haas_effect_decorrelation():
    input_signal = np.ones(435)
    haas = HaasEffect(sample_rate_hz=44100, delay_time_seconds=0.02)
    output_signal = haas(input_signal)

    assert output_signal.shape[0] == 1317  # 435 + 0.02 * 44100
    assert output_signal.shape[1] == 2
    assert 0.0 != np.sum(output_signal)


def test__heterogeneous_signal_chain():
    fs = 44100
    input_signal = np.zeros(1000)
    chain = (
        SignalChain(sample_rate_hz=fs)
        .velvet_noise(
            duration_seconds=0.03,
            num_impulses=30,
            width=0.5,
        )
        .haas_effect(
            delay_time_seconds=0.0197,
            delayed_channel=1,
            mode='LR',
        )
        .haas_effect(
            delay_time_seconds=0.0096,
            delayed_channel=1,
            mode='MS',
        )
        .white_noise(duration_seconds=0.03, width=0.5)
        .stateless(
            convolve_velvet_noise,
            generate_velvet_noise(duration_seconds=0.03, num_impulses=30),
        )
    )
    output_signal = chain(input_signal)

    assert output_signal.shape[0] > 1000
    assert output_signal.shape[1] == 2


def test__velvet_noise_generation():
    vnd = VelvetNoise(sample_rate_hz=44100, seed=1)
    sequences = vnd._generate()

    assert sequences == vnd._velvet_noise

    vnd = VelvetNoise(sample_rate_hz=44100, seed=2)

    assert sequences != vnd._velvet_noise

    vnd = VelvetNoise(sample_rate_hz=44100, log_distribution_strength=0.0, seed=1)

    assert sequences != vnd._velvet_noise


def test__velvet_noise_generation_equality():
    vnd = VelvetNoise(
        duration_seconds=0.03,
        num_impulses=30,
        num_outs=2,
        sample_rate_hz=44100,
        segment_envelope=(0.85, 0.55, 0.35, 0.2),
        log_distribution_strength=1.0,
        seed=1,
    )
    FIR = (
        generate_velvet_noise(
            duration_seconds=0.03,
            num_impulses=30,
            num_outs=2,
            sample_rate_hz=44100,
            segment_envelope=(0.85, 0.55, 0.35, 0.2),
            log_distribution_strength=1.0,
            seed=1,
        ),
    )

    assert np.allclose(vnd.FIR, FIR, atol=1e-6)


def test__velvet_noise_properties():
    fs = 44100
    vnd = VelvetNoise(
        duration_seconds=0.03, num_impulses=30, sample_rate_hz=fs, num_outs=2
    )

    assert vnd.density == 1000

    vnd = VelvetNoise(
        sample_rate_hz=44100,
        duration_seconds=0.055,
        num_impulses=45,
    )

    assert 818.19 > vnd.density > 818.18
    assert vnd.FIR.shape == (2426, 2)
    assert len(vnd.FIR[:, 0][vnd.FIR[:, 0] != 0.0]) == 45

    with pytest.raises(ValueError):
        vnd = VelvetNoise(duration_seconds=0.03, num_impulses=700, sample_rate_hz=fs)


def test__velvet_noise_decorrelation():
    vnd = VelvetNoise(sample_rate_hz=44100)
    input_signal = np.zeros(1000)
    output_signal = vnd(input_signal)
    assert output_signal.shape[0] == 1000
    assert output_signal.shape[1] == 2


def test__velvet_noise_decorrelation_15():
    vnd = VelvetNoise(num_impulses=15, duration_seconds=0.5, sample_rate_hz=44100)
    input_signal = np.zeros(1000)
    output_signal = vnd(input_signal)

    assert output_signal.shape[0] == 1000
    assert output_signal.shape[1] == 2


def test__velvet_noise_decorrelation_segment_envelope():
    vnd = VelvetNoise(
        num_impulses=15,
        duration_seconds=0.5,
        sample_rate_hz=44100,
        segment_envelope=(1.0, 0.5, 0.25),
    )
    input_signal = np.zeros(1000)
    output_signal = vnd(input_signal)

    assert output_signal.shape[0] == 1000
    assert output_signal.shape[1] == 2

    vnd.segment_envelope = ()
    output_signal = vnd(input_signal)

    assert output_signal.shape[0] == 1000
    assert output_signal.shape[1] == 2

    vnd.segment_envelope = [1.0] * 1000
    output_signal = vnd(input_signal)

    assert output_signal.shape[0] == 1000
    assert output_signal.shape[1] == 2


def test__velvet_noise_generated_once():
    vnd = VelvetNoise(sample_rate_hz=44100, seed=1)
    input_signal = np.zeros(1000)
    vns_1 = vnd._velvet_noise
    _ = vnd(input_signal)
    _ = vnd.FIR
    vns_2 = vnd.velvet_noise

    assert np.array_equal(vns_1, vns_2)


def test__convolve_velvet_noise_equality():
    input_signal = np.random.random((10000, 2))
    output_signal_1 = convolve_velvet_noise(
        input_signal,
        generate_velvet_noise(
            duration_seconds=0.03,
            num_impulses=30,
            num_outs=2,
            sample_rate_hz=44100,
            segment_envelope=(0.85, 0.55, 0.35, 0.2),
            log_distribution_strength=1.0,
            seed=1,
        ),
    )
    output_signal_2 = VelvetNoise(
        duration_seconds=0.03,
        num_impulses=30,
        num_outs=2,
        sample_rate_hz=44100,
        segment_envelope=(0.85, 0.55, 0.35, 0.2),
        log_distribution_strength=1.0,
        seed=1,
    ).convolve(input_signal)

    assert output_signal_1.shape == output_signal_2.shape
    assert np.allclose(output_signal_1, output_signal_2, atol=1e-6)


def test__white_noise_decorrelation():
    input_signal = np.random.random(4350)
    output_signal = WhiteNoise(sample_rate_hz=44100, duration_seconds=0.03)(
        input_signal
    )

    assert output_signal.shape[0] == 4350
    assert output_signal.shape[1] == 2
