import numpy as np
import unittest
from VNDecorrelate.utils import plot

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