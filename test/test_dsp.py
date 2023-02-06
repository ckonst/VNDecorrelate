# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:13:07 2022

@author: Christian Konstantinov
"""

import numpy as np
import unittest
from utils import dsp

class DSPTestCase(unittest.TestCase):

    def test_apply_stereo_width(self):
        x1 = np.random.uniform(size=(100,2))
        y1 = dsp.apply_stereo_width(x1, 1.0)
        self.assertTrue(np.sum(dsp.LR_to_MS(y1)[:, 0]) == 0.0)
        self.assertTrue(np.sum(dsp.LR_to_MS(y1)[:, 1]) != 0.0)
        x2 = np.random.uniform(size=(100,2))
        y2 = dsp.apply_stereo_width(x2, 0.0)
        self.assertTrue(np.sum(dsp.LR_to_MS(y2)[:, 1]) == 0.0)
        self.assertTrue(np.sum(dsp.LR_to_MS(y2)[:, 0]) != 0.0)
        x3 = np.full((100, 2), 100)
        y3 = dsp.apply_stereo_width(x3, 0.5)
        self.assertTrue(np.sum(y3) == 100 * 100)
        x4 = np.full((100, 2), 100)
        y4 = dsp.apply_stereo_width(x4, 0.25)
        self.assertTrue(not np.array_equal(x4, y4))

    def test_encode_signal_to_side_channel(self):
        x1 = np.array([])
        x2 = np.array([])
        with self.assertRaises(ValueError):
            dsp.encode_signal_to_side_channel(x1, x2)
        x1 = np.zeros((100, 2))
        with self.assertRaises(ValueError):
            dsp.encode_signal_to_side_channel(x1, x2)
        x2 = np.zeros((120, 2))
        with self.assertRaises(ValueError):
            dsp.encode_signal_to_side_channel(x1, x2)
        x2 = np.column_stack((np.zeros(100,), np.full((100,), 100)))
        y1 = dsp.encode_signal_to_side_channel(x1, x2)
        y1 = dsp.LR_to_MS(y1)
        self.assertTrue(np.array_equal(y1[:, 1], (x2[:, 0] - x2[:, 1]) / 2))
        x2 = np.full((100,2), 100)
        y1 = dsp.encode_signal_to_side_channel(x1, x2)
        y1 = dsp.LR_to_MS(y1)
        self.assertTrue(np.array_equal(y1[:, 1], np.zeros((100,))))

    def test_to_float32(self):
        x = np.array([1,2,3,4], dtype=np.int32)
        self.assertTrue(dsp.to_float32(x).dtype == np.float32)
        x = np.array([1,2,3,4], dtype=np.float32)
        self.assertTrue(dsp.to_float32(x).dtype == np.float32)

    def test_peak_normalize(self):
        x1 = np.random.normal(loc=0.0, scale=100.0, size=(100,100))
        x2 = np.zeros((10_000))
        x3 = np.random.uniform(low=2.0, high=10.0, size=(50_000, 2))
        x4 = np.random.uniform(size=(100, 100))
        dsp.peak_normalize(x1)
        self.assertTrue(np.max(np.abs(x1)) <= 1.0)
        dsp.peak_normalize(x2)
        self.assertTrue(np.max(np.abs(x2)) <= 1.0)
        dsp.peak_normalize(x3)
        self.assertTrue(np.max(np.abs(x3)) <= 1.0)
        dsp.peak_normalize(x4)
        self.assertTrue(np.max(np.abs(x1)) <= 1.0)

    def test_rms_normalize(self):
        x1 = np.random.normal(loc=0.0, scale=100.0, size=(100,100))
        x2 = np.zeros((10_000))
        x3 = np.random.uniform(low=2.0, high=10.0, size=(50_000, 2))
        x4 = np.random.uniform(size=(100, 100))
        sum_ = np.sum(x4)
        dsp.rms_normalize(x1, x4)
        self.assertTrue(sum_ != np.sum(x4))
        dsp.rms_normalize(x2, x4)
        self.assertTrue(sum_ != np.sum(x4))
        dsp.rms_normalize(x3, x4)
        self.assertTrue(sum_ != np.sum(x4))

    def test_mono_to_stereo(self):
        x = np.array([1,2,3,4])
        y = dsp.mono_to_stereo(x)
        y_truth = np.array([[1, 1],[2, 2], [3, 3], [4, 4]])
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.array([])
        y = dsp.mono_to_stereo(x)
        y_truth = np.zeros((0, 2))
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.array([[1, 1],[2, 2], [3, 3], [4, 4]])
        with self.assertRaises(ValueError):
            dsp.mono_to_stereo(x)

    def test_stereo_to_mono(self):
        x = np.array([[1, 1],[2, 2], [3, 3], [4, 4]])
        y = dsp.stereo_to_mono(x)
        y_truth = np.array([1,2,3,4])
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.zeros((0, 2))
        y = dsp.stereo_to_mono(x)
        y_truth = np.array([])
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.array([])
        with self.assertRaises(ValueError):
            dsp.stereo_to_mono(x)
        x = np.array([1,2,3,4])
        with self.assertRaises(ValueError):
            dsp.stereo_to_mono(x)

    def test_LR_to_MS(self):
        x = np.array([[1, 1],[2, 2], [3, 3], [4, 4]])
        y = dsp.LR_to_MS(x)
        y_truth = np.array([[1, 0],[2, 0], [3, 0], [4, 0]])
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.array([[1, 2],[2, 4], [3, 6], [4, 8]])
        y = dsp.LR_to_MS(x)
        y_truth = np.array([[1.5, -0.5],[3, -1], [4.5, -1.5], [6, -2]])
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.array([])
        with self.assertRaises(ValueError):
            dsp.stereo_to_mono(x)
        x = np.array([1,2,3,4])
        with self.assertRaises(ValueError):
            dsp.stereo_to_mono(x)

    def test_MS_to_LR(self):
        x = np.array([[1, 1],[2, 2], [3, 3], [4, 4]])
        y = dsp.MS_to_LR(x)
        y_truth = np.array([[2, 0],[4, 0], [6, 0], [8, 0]])
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.array([[1, 2],[2, 4], [3, 6], [4, 8]])
        y = dsp.MS_to_LR(x)
        y_truth = np.array([[3, -1],[6, -2], [9, -3], [12, -4]])
        self.assertTrue(np.array_equal(y, y_truth))
        x = np.array([])
        with self.assertRaises(ValueError):
            dsp.stereo_to_mono(x)
        x = np.array([1,2,3,4])
        with self.assertRaises(ValueError):
            dsp.stereo_to_mono(x)
