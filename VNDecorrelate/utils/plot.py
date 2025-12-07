import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_correlogram(
    correlogram: NDArray,
    lag_seconds: float,
    time_seconds: int,
    title: str = 'Cross Correlogram',
) -> None:
    plt.figure(figsize=(15, 6))
    plt.imshow(
        np.abs(correlogram),
        extent=[-lag_seconds, lag_seconds, time_seconds, 0],
        aspect='auto',
        cmap='Blues',
        origin='upper',
    )
    plt.colorbar(label='Correlation Coefficient (Normalized)')
    plt.xlabel('Lag (s)')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.savefig(f'./tests/plots/{title}.png')
    plt.close()


def plot_signal(input_sig: NDArray, title: str = 'Signal') -> None:
    """Plot the time domain input signal."""
    plt.figure()
    plt.plot(input_sig)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.savefig(f'./tests/plots/{title}.png')
    plt.close()


def plot_spectrogram(
    spectrogram: NDArray,
    title: str = 'Spectrogram',
) -> None:
    """Plot the spectrogram of the input signal."""
    plt.figure(figsize=(15, 6))
    plt.imshow(
        10 * np.log10(spectrogram + 1e-10),
        aspect='auto',
        cmap='Blues',
        origin='lower',
    )
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title(title)
    plt.savefig(f'./tests/plots/{title}.png')
    plt.close()
