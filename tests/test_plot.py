import unittest
from unittest import TestCase

import numpy as np

from VNDecorrelate.utils import plot


class PlotTestCase(TestCase):
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
