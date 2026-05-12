import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from vndecorrelate.utils.dsp import polar_coordinates


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


def plot_signal(input_signal: NDArray, title: str = 'Signal') -> None:
    """Plot the time domain input signal."""
    plt.figure()
    plt.plot(input_signal)
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


def plot_polar_sample(
    input_signal: NDArray,
    alpha: float = 0.05,
    dot_size: float = 0.5,
    title: str = 'Vectorscope',
    heatmap: bool = True,
) -> None:
    """Plot the polar sample vectorscope of the input signal."""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    ax.set_theta_zero_location('N')  # Top is 0
    ax.set_theta_direction(-1)  # Clockwise

    ax.set_title(title, color='black', fontsize=12)

    ax.set_xticks(np.linspace(-np.pi / 2, np.pi / 2, 5))
    ax.set_xticklabels(['L', '-45°', 'C', '45°', 'R'])

    ax.set_rticks(())
    ax.set_yticklabels(())

    left, right = input_signal.T
    r, theta, _ = polar_coordinates(left, right)

    if heatmap:
        bins = (
            np.linspace(-np.pi / 2, np.pi / 2, 256, endpoint=True),
            np.linspace(0, 1, 256),
        )
        counts, xedges, yedges = np.histogram2d(
            theta, r, bins=bins, range=[[-1, 1], [0, 1]]
        )
        A, R = np.meshgrid(xedges, yedges)

        gamma = 0.4
        heatmap_corrected = np.power(counts / counts.max(), gamma)

        ax.pcolormesh(A, R, heatmap_corrected.T, cmap='viridis')
    else:
        ax.scatter(theta, r, alpha=alpha, s=dot_size, cmap='blues')

    plt.savefig(f'./tests/plots/{title}.png')
    plt.close()
