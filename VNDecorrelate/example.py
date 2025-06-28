import scipy.io.wavfile as wavfile

from VNDecorrelate.decorrelation import SignalChain


def main() -> None:
    fs, guitar = wavfile.read('audio/guitar.wav')
    chain = (
        SignalChain()
        .velvet_noise(
            fs=fs,
            duration=0.03,
            num_impulses=30,
            seed=1,
            use_log_distribution=True,
        )
        .haas_effect(
            delay_time_seconds=0.02,
            fs=fs,
            delayed_channel=1,
            mode='LR',
        )
    )
    output_sig = chain.decorrelate(guitar)
    wavfile.write('audio/guitar_dec.wav', fs, output_sig)

    fs, pop_shuffle = wavfile.read('audio/pop_shuffle.wav')
    output_sig = chain.decorrelate(pop_shuffle)
    wavfile.write('audio/pop_shuffle_dec.wav', fs, output_sig)

    fs, viola = wavfile.read('audio/viola.wav')
    output_sig = chain.decorrelate(viola)
    wavfile.write('audio/viola_dec.wav', fs, output_sig)

    fs, vocal = wavfile.read('audio/vocal.wav')
    output_sig = chain.decorrelate(vocal)
    wavfile.write('audio/vocal_dec.wav', fs, output_sig)


if __name__ == '__main__':
    main()
