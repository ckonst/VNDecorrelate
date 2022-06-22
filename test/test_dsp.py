# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:13:07 2022

@author: Christian Konstantinov
"""

import numpy as np
import unittest
from utils import dsp

class DSPTestCase(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.x1 = np.random.normal(loc=0.0, scale=100.0, size=(100,100))
        cls.x2 = np.zeros((10_000))
        cls.x3 = np.random.uniform(low=2.0, high=10.0, size=(50_000, 2))

    def test_rms_normalize(self):
        x4 = np.random.uniform(size=(100, 100))
        sum_ = np.sum(x4)
        dsp.rms_normalize(self.x1, x4)
        assert sum_ != np.sum(x4)
        dsp.rms_normalize(self.x2, x4)
        assert sum_ != np.sum(x4)
        dsp.rms_normalize(self.x3, x4)
        assert sum_ != np.sum(x4)

    def test_peak_normalize(self):
        x4 = np.random.uniform(size=(100, 100))
        dsp.peak_normalize(self.x1)
        dsp.peak_normalize(self.x2)
        dsp.peak_normalize(self.x3)
        dsp.peak_normalize(x4)
        assert True

    def test_LR_to_MS(self):
        pass

    def test_MS_to_LR(self):
        pass
