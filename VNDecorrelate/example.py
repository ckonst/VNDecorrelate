from VNDecorrelate.decorrelation import SignalChain
import scipy.io.wavfile as wavfile


def main() -> None:
    fs, guitar = wavfile.read('audio/guitar.wav')
    chain = (
        SignalChain(fs=fs, num_ins=1, num_outs=2)
        .velvet_noise(fs=fs, duration=0.03, num_impulses=30, seed=1, use_log_distribution=True)
        .haas_effect(0.02, fs=fs, channel=1, mode='LR')
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
