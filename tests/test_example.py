import scipy.io.wavfile as wavfile

from vndecorrelate.decorrelation import SignalChain


# @pytest.mark.skip
def test_example():
    fs, guitar = wavfile.read('audio/guitar.wav')

    chain = (
        SignalChain(sample_rate_hz=fs)
        .velvet_noise(
            duration_seconds=0.03,
            num_impulses=30,
            seed=1,
            use_log_distribution=True,
        )
        .haas_effect(
            delay_time_seconds=0.02,
            delayed_channel=1,
            mode='LR',
        )
    )

    output_signal = chain(guitar)
    wavfile.write('audio/guitar_dec.wav', fs, output_signal)

    fs, pop_shuffle = wavfile.read('audio/pop_shuffle.wav')
    output_signal = chain(pop_shuffle)
    wavfile.write('audio/pop_shuffle_dec.wav', fs, output_signal)

    fs, viola = wavfile.read('audio/viola.wav')
    output_signal = chain(viola)
    wavfile.write('audio/viola_dec.wav', fs, output_signal)

    fs, vocal = wavfile.read('audio/vocal.wav')
    output_signal = chain(vocal)
    wavfile.write('audio/vocal_dec.wav', fs, output_signal)

    assert True
