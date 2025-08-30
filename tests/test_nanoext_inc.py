import unittest
import numpy as np

from nanoext import zscore_batch_rb_inc, corr_pairs_rb_inc


class TestIncrementalPaths(unittest.TestCase):
    def setUp(self):
        self.window = 64
        self.lookback = 32
        # Build simple sequences
        x = np.arange(self.window, dtype=np.float64)
        y_same = np.copy(x)
        y_inv = -x
        # price_rb with two symbols
        self.price_same = np.vstack([x, y_same])
        self.price_inv = np.vstack([x, y_inv])
        # widx points to next write (0 -> last sample at index window-1)
        self.widx = np.array([0, 0], dtype=np.int32)
        self.pairs = np.array([0, 1], dtype=np.int32)  # one pair

    def test_zscore_identical_series_zero_signal(self):
        thresholds = np.array([1e9], dtype=np.float64)
        sums = np.zeros(1, dtype=np.float64)
        sumsq = np.zeros(1, dtype=np.float64)
        out = zscore_batch_rb_inc(self.price_same, self.widx, self.pairs, self.lookback,
                                  thresholds, sums, sumsq, 0)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(int(out[0]), 0)

    def test_corr_identical_series_near_one(self):
        sx = np.zeros(1, dtype=np.float64)
        sy = np.zeros(1, dtype=np.float64)
        sxx = np.zeros(1, dtype=np.float64)
        syy = np.zeros(1, dtype=np.float64)
        sxy = np.zeros(1, dtype=np.float64)
        corr = corr_pairs_rb_inc(self.price_same, self.widx, self.pairs, self.lookback,
                                 sx, sy, sxx, syy, sxy, 0)
        self.assertEqual(corr.shape[0], 1)
        self.assertGreater(corr[0], 0.99)

    def test_corr_inverted_series_near_minus_one(self):
        sx = np.zeros(1, dtype=np.float64)
        sy = np.zeros(1, dtype=np.float64)
        sxx = np.zeros(1, dtype=np.float64)
        syy = np.zeros(1, dtype=np.float64)
        sxy = np.zeros(1, dtype=np.float64)
        corr = corr_pairs_rb_inc(self.price_inv, self.widx, self.pairs, self.lookback,
                                 sx, sy, sxx, syy, sxy, 0)
        self.assertEqual(corr.shape[0], 1)
        self.assertLess(corr[0], -0.99)


if __name__ == '__main__':
    unittest.main()

