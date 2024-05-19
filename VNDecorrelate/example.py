from VNDecorrelate.decorrelation import SignalChain
import scipy.io.wavfile as wavfile

def main() -> None:
    fs, sig_float32 = wavfile.read('audio/guitar.wav')
    chain = (
        SignalChain(fs=fs, num_ins=1, num_outs=2)
        .velvet_noise(fs=fs, duration=0.06, num_impulses=60, seed=1, use_log_distribution=False)
        .haas_effect(0.0197, fs=fs, channel=1, mode='LR')
        .haas_effect(0.0096, fs=fs, channel=1, mode='MS')
    )
    output_sig = chain(sig_float32)
    wavfile.write('audio/guitar_dec.wav', fs, output_sig)


if __name__ == '__main__':
    main()
    