from unittest import TestCase

import numpy as np

from VNDecorrelate.decorrelation import HaasEffect, SignalChain, VelvetNoise


class DecorrelationTestCase(TestCase):
    def test_signal_chain(self):
        fs = 44100
        input_sig = np.zeros(1000)
        chain = (
            SignalChain(sample_rate_hz=fs)
            .velvet_noise(
                duration=0.03,
                num_impulses=30,
                width=1.0,
            )
            .haas_effect(
                delay_time_seconds=0.0197,
                delayed_channel=1,
                mode='LR',
            )
            .haas_effect(
                delay_time_seconds=0.0096,
                delayed_channel=1,
                mode='MS',
            )
        )
        output_sig = chain(input_sig)
        self.assertTrue(output_sig.shape[0] > 1000)
        self.assertTrue(output_sig.shape[1] == 2)

    def test_velvet_noise_generation(self):
        vnd = VelvetNoise(sample_rate_hz=44100, seed=1)
        sequences = vnd._generate()
        self.assertTrue(sequences == vnd._vn_sequence)
        vnd = VelvetNoise(sample_rate_hz=44100, seed=2)
        self.assertFalse(sequences == vnd._vn_sequence)
        vnd = VelvetNoise(sample_rate_hz=44100, use_log_distribution=False, seed=1)
        self.assertFalse(sequences == vnd._vn_sequence)

    def test_properties(self):
        fs = 44100
        vnd = VelvetNoise(duration=0.03, num_impulses=30, sample_rate_hz=fs, num_outs=2)
        self.assertTrue(vnd.density == 1000)
        vnd = VelvetNoise(
            sample_rate_hz=44100,
            duration=0.055,
            num_impulses=45,
        )
        self.assertTrue(818.19 > vnd.density > 818.18)
        self.assertTrue(vnd.FIR.shape == (2425, 2))

    def test_decorrelation(self):
        vnd = VelvetNoise(sample_rate_hz=44100)
        input_sig = np.zeros(1000)
        output_sig = vnd(input_sig)
        self.assertTrue(output_sig.shape[0] == 1000)
        self.assertTrue(output_sig.shape[1] == 2)

        input_sig_2 = np.zeros(435)
        haas = HaasEffect(sample_rate_hz=44100)
        output_sig_2 = haas(input_sig_2)
        self.assertTrue(output_sig_2.shape[0] > 435)
        self.assertTrue(output_sig_2.shape[1] == 2)
