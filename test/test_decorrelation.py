# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 21:15:31 2023

@author: Christian Konstantinov
"""

import numpy as np
import unittest
from decorrelation import SignalChain, VelvetNoise, HaasEffect

class DecorrelationTestCase(unittest.TestCase):

    def test_signal_chain(self):
        fs = 44100
        input_sig = np.zeros(1000)
        chain = (
            SignalChain(fs=fs, num_ins=1, num_outs=2)
                .velvet_noise(fs=fs, duration=0.03, num_impulses=30, width=1.0)
                .haas_effect(0.0197, fs=fs, channel=1, mode='LR')
                .haas_effect(0.0096, fs=fs, channel=1, mode='MS')
        )
        output_sig = chain(input_sig)
        self.assertTrue(output_sig.shape[0] > 1000)
        self.assertTrue(output_sig.shape[1] == 2)

    def test_velvet_noise_generation(self):
        vnd = VelvetNoise(seed=1)
        sequences = vnd._generate()
        self.assertTrue(sequences == vnd._vn_sequences)
        vnd.seed = 2
        vnd.regenerate()
        self.assertFalse(sequences == vnd._vn_sequences)

    def test_properties(self):
        fs = 44100
        vnd = VelvetNoise(duration = 0.03, num_impulses = 30, fs=fs, num_outs=2)
        self.assertTrue(vnd.density == 1000)
        vnd.duration = 0.055
        vnd.num_impulses = 45
        vnd.regenerate
        self.assertTrue(818.19 > vnd.density > 818.18)
        self.assertTrue(vnd.FIR.shape == (2425, 2))

    def test_decorrelation(self):
        vnd = VelvetNoise()
        input_sig = np.zeros(1000)
        output_sig = vnd(input_sig)
        self.assertTrue(output_sig.shape[0] == 1000)
        self.assertTrue(output_sig.shape[1] == 2)

        input_sig_2 = np.zeros(435)
        haas = HaasEffect()
        output_sig_2 = haas(input_sig_2)
        self.assertTrue(output_sig_2.shape[0] > 435)
        self.assertTrue(output_sig_2.shape[1] == 2)
