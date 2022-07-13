# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:13:07 2022

@author: Christian Konstantinov
"""

import numpy as np
import unittest
from utils import plot

class PlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_plots_basic(self):
        # simply run the main function to ensure no errors.
        plot.main()
        self.assertTrue(True)

    def test_plot_signal(self):
        x = np.random.uniform(size=100)
        plot.plot_signal(x)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()