import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from numpy.typing import NDArray

from VNDecorrelate.decorrelation import VelvetNoise

# TODO: Plot Autocorrelogram and Cross Correlogram of Sine-sweep signal


def plot_signal(input_sig: NDArray, title: str = 'Signal') -> None:
    """Plot the time domain input signal."""
    plt.figure()
    plt.plot(input_sig)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(title)


def main():
    """Plot input and output signals, and velvet noise sequences."""
    VNS_DURATION = 0.03  # duration of VNS in seconds
    M = 30  # number of impulses

    sample_rate, sig_float32 = wavfile.read('audio/guitar.wav')
    vnd = VelvetNoise(
        sample_rate_hz=sample_rate,
        duration=VNS_DURATION,
        num_impulses=M,
        num_outs=2,
    )
    result = vnd.decorrelate(sig_float32)
    vns = vnd.FIR

    plot_signal(vns[:, 0], title='Velvet Noise Sequence L')

    plot_signal(vns[:, 1], title='Velvet Noise Sequence R')

    plot_signal(sig_float32, title='Input')

    plot_signal(result, title='Output')


if __name__ == '__main__':
    main()
