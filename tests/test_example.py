import pytest
import scipy.io.wavfile as wavfile

from tests import IN_GHA
from vndecorrelate.decorrelation import SignalChain


@pytest.mark.skipif(IN_GHA, reason='Skipping in CI')
def test_example(
    guitar_signal,
    drums_signal,
    viola_signal,
    vocal_signal,
):
    fs, guitar = guitar_signal

    # Intentionally exaggerated and phasy for demonstration purposes:
    # Filter one channel with Velvet Noise, and sample (Haas) delay the other.
    chain = (
        SignalChain(sample_rate_hz=fs)
        .velvet_noise(
            duration_seconds=0.02,
            num_impulses=30,
            seed=1,
            log_distribution_strength=1.0,
            mode='MS',
            filtered_channels=(0, 1),
        )
        .haas_effect(
            delay_time_seconds=0.02,
            delayed_channel=1,
            mode='LR',
        )
    )

    output_signal = chain(guitar)
    wavfile.write('audio/guitar_decorrelated.wav', fs, output_signal)

    fs, drums = drums_signal
    output_signal = chain(drums)
    wavfile.write('audio/drums_decorrelated.wav', fs, output_signal)

    fs, viola = viola_signal
    output_signal = chain(viola)
    wavfile.write('audio/viola_decorrelated.wav', fs, output_signal)

    fs, vocal = vocal_signal
    output_signal = chain(vocal)
    wavfile.write('audio/vocal_decorrelated.wav', fs, output_signal)

    assert True
