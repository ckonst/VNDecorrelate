import pytest
import scipy.io.wavfile as wavfile

from tests import IN_GHA
from vndecorrelate.decorrelation import SignalChain


@pytest.mark.skipif(IN_GHA, reason='Skipping in CI')
def test_example(
    guitar_signal,
    pop_shuffle_signal,
    viola_signal,
    vocal_signal,
):
    fs, guitar = guitar_signal

    chain = (
        SignalChain(sample_rate_hz=fs)
        .velvet_noise(
            duration_seconds=0.03,
            num_impulses=30,
            seed=1,
            log_distribution_strength=1.0,
        )
        .haas_effect(
            delay_time_seconds=0.02,
            delayed_channel=1,
            mode='LR',
        )
    )

    output_signal = chain(guitar)
    wavfile.write('audio/guitar_dec.wav', fs, output_signal)

    fs, pop_shuffle = pop_shuffle_signal
    output_signal = chain(pop_shuffle)
    wavfile.write('audio/pop_shuffle_dec.wav', fs, output_signal)

    fs, viola = viola_signal
    output_signal = chain(viola)
    wavfile.write('audio/viola_dec.wav', fs, output_signal)

    fs, vocal = vocal_signal
    output_signal = chain(vocal)
    wavfile.write('audio/vocal_dec.wav', fs, output_signal)

    assert True
