from unittest import TestCase

import numpy as np

from VNDecorrelate.decorrelation import HaasEffect, SignalChain, VelvetNoise
from VNDecorrelate.utils.dsp import generate_velvet_noise


class DecorrelationTestCase(TestCase):
    def test__heterogeneous_signal_chain(self):
        fs = 44100
        input_sig = np.zeros(1000)
        chain = (
            SignalChain(sample_rate_hz=fs)
            .velvet_noise(
                duration_seconds=0.03,
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
        self.assertGreater(output_sig.shape[0], 1000)
        self.assertEqual(output_sig.shape[1], 2)

    def test__velvet_noise_generation(self):
        vnd = VelvetNoise(sample_rate_hz=44100, seed=1)
        sequences = vnd._generate()
        self.assertEqual(sequences, vnd._velvet_noise)
        vnd = VelvetNoise(sample_rate_hz=44100, seed=2)
        self.assertNotEqual(sequences, vnd._velvet_noise)
        vnd = VelvetNoise(sample_rate_hz=44100, use_log_distribution=False, seed=1)
        self.assertNotEqual(sequences, vnd._velvet_noise)

    def test__velvet_noise_generation_equality(self):
        vnd = VelvetNoise(
            duration_seconds=0.03,
            num_impulses=30,
            num_outs=2,
            sample_rate_hz=44100,
            segment_envelope=(0.85, 0.55, 0.35, 0.2),
            use_log_distribution=True,
            seed=1,
        )
        self.assertTrue(
            np.allclose(
                vnd.FIR,
                generate_velvet_noise(
                    duration_seconds=0.03,
                    num_impulses=30,
                    num_outs=2,
                    sample_rate_hz=44100,
                    segment_envelope=(0.85, 0.55, 0.35, 0.2),
                    use_log_distribution=True,
                    seed=1,
                ),
            )
        )

    def test__velvet_noise_properties(self):
        fs = 44100
        vnd = VelvetNoise(
            duration_seconds=0.03, num_impulses=30, sample_rate_hz=fs, num_outs=2
        )
        self.assertEqual(vnd.density, 1000)
        vnd = VelvetNoise(
            sample_rate_hz=44100,
            duration_seconds=0.055,
            num_impulses=45,
        )
        self.assertTrue(818.19 > vnd.density > 818.18)
        self.assertEqual(vnd.FIR.shape, (2426, 2))
        self.assertEqual(len(vnd.FIR[:, 0][vnd.FIR[:, 0] != 0.0]), 45)
        with self.assertRaises(ValueError):
            vnd = VelvetNoise(
                duration_seconds=0.03, num_impulses=700, sample_rate_hz=fs
            )

    def test__velvet_noise_decorrelation(self):
        vnd = VelvetNoise(sample_rate_hz=44100)
        input_sig = np.zeros(1000)
        output_sig = vnd(input_sig)
        self.assertEqual(output_sig.shape[0], 1000)
        self.assertEqual(output_sig.shape[1], 2)

    def test__velvet_noise_decorrelation_15(self):
        vnd = VelvetNoise(num_impulses=15, duration_seconds=0.5, sample_rate_hz=44100)
        input_sig = np.zeros(1000)
        output_sig = vnd(input_sig)
        self.assertEqual(output_sig.shape[0], 1000)
        self.assertEqual(output_sig.shape[1], 2)

    def test__velvet_noise_decorrelation_segment_envelope(self):
        vnd = VelvetNoise(
            num_impulses=15,
            duration_seconds=0.5,
            sample_rate_hz=44100,
            segment_envelope=(1.0, 0.5, 0.25),
        )
        input_sig = np.zeros(1000)
        output_sig = vnd(input_sig)
        self.assertEqual(output_sig.shape[0], 1000)
        self.assertEqual(output_sig.shape[1], 2)

        vnd = VelvetNoise(
            num_impulses=15,
            duration_seconds=0.5,
            sample_rate_hz=44100,
            segment_envelope=(),
        )
        input_sig = np.zeros(1000)
        output_sig = vnd(input_sig)
        self.assertEqual(output_sig.shape[0], 1000)
        self.assertEqual(output_sig.shape[1], 2)

    def test__velvet_noise_generated_once(self):
        vnd = VelvetNoise(sample_rate_hz=44100, seed=1)
        input_sig = np.zeros(1000)
        vns_1 = vnd._velvet_noise
        _ = vnd(input_sig)
        _ = vnd.FIR
        vns_2 = vnd.velvet_noise
        self.assertTrue(np.array_equal(vns_1, vns_2))

    def test__haas_effect_decorrelation(self):
        input_sig_2 = np.zeros(435)
        haas = HaasEffect(sample_rate_hz=44100)
        output_sig_2 = haas(input_sig_2)
        self.assertGreater(output_sig_2.shape[0], 435)
        self.assertEqual(output_sig_2.shape[1], 2)
