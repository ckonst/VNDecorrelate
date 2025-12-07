from unittest import TestCase

import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal

from VNDecorrelate.decorrelation import SignalChain, VelvetNoise
from VNDecorrelate.utils import dsp, plot


class PlotTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_plots_basic(self):
        # simply run the main function to ensure no errors.
        sample_rate, sig_float32 = wavfile.read('audio/viola.wav')
        vnd = VelvetNoise(
            sample_rate_hz=sample_rate,
            duration=0.03,
            num_impulses=30,
            num_outs=2,
            use_log_distribution=True,
        )
        result = vnd.decorrelate(sig_float32)
        vns = vnd.FIR

        plot.plot_signal(vns[:, 0], title='Velvet Noise Sequence L')

        plot.plot_signal(vns[:, 1], title='Velvet Noise Sequence R')

        plot.plot_signal(sig_float32, title='Input')

        plot.plot_signal(result, title='Output')

        self.assertTrue(True)

    def test_plot_signal(self):
        x = np.random.uniform(size=100)
        plot.plot_signal(x)
        self.assertTrue(True)

    def test_plot_correlogram(self):
        fs = 16000  # 16 kHz sample rate
        sine_sweep = dsp.generate_sine_sweep(
            start_freq_hz=20,
            end_freq_hz=8000,
            duration_seconds=9,
            sample_rate_hz=fs,
        )
        plot.plot_spectrogram(
            signal.spectrogram(sine_sweep, fs=fs)[-1], title='Sine Sweep Signal'
        )
        correlogram = dsp.cross_correlogram(
            sine_sweep,
            sine_sweep,
            sample_rate_hz=fs,
            max_lag_seconds=0.02,
            window_size_seconds=0.02,
            stride_seconds=0.01,
        )
        plot.plot_correlogram(
            correlogram,
            lag_seconds=0.02,
            time_seconds=9,
            title='Sine-sweep Auto Correlogram',
        )
        self.assertTrue(True)

    def test_plot_correlogram__from_file(self):
        fs, guitar = wavfile.read('audio/guitar.wav')

        chain = (
            SignalChain(sample_rate_hz=fs)
            .velvet_noise(
                duration=0.03,
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

        output_sig = chain(guitar)
        correlogram = dsp.cross_correlogram(
            output_sig[:, 0],
            output_sig[:, 1],
            sample_rate_hz=fs,
            max_lag_seconds=0.02,
            window_size_seconds=0.02,
            stride_seconds=0.01,
        )
        plot.plot_correlogram(
            correlogram,
            lag_seconds=0.02,
            time_seconds=9,
            title='Guitar Output Cross Correlogram',
        )
        self.assertTrue(True)
