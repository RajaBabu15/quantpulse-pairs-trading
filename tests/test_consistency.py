import unittest
import numpy as np

from nanoext import zscore_batch_rb_inc, corr_pairs_rb_inc


def full_recompute_zscores(price_rb, widx, pairs, lookback, thresholds):
    n_pairs = len(pairs)//2
    out = np.zeros(n_pairs, dtype=np.int8)
    n_symbols, window = price_rb.shape
    for i in range(n_pairs):
        a, b = pairs[2*i], pairs[2*i+1]
        wa, wb = widx[a], widx[b]
        sa = wa - lookback
        if sa < 0: sa += window
        sb = wb - lookback
        if sb < 0: sb += window
        ia, ib = sa, sb
        s_sum = 0.0; s_sumsq = 0.0
        for k in range(lookback):
            p1 = price_rb[a, ia]
            p2 = price_rb[b, ib]
            sp = p1 - p2
            s_sum += sp
            s_sumsq += sp*sp
            ia += 1; ib += 1
            if ia == window: ia = 0
            if ib == window: ib = 0
        mean = s_sum / lookback
        var = s_sumsq / lookback - mean*mean
        if var <= 0.0:
            out[i] = 0
            continue
        stdv = np.sqrt(var)
        cur_a = wa - 1
        cur_b = wb - 1
        if cur_a < 0: cur_a += window
        if cur_b < 0: cur_b += window
        z = (price_rb[a, cur_a] - price_rb[b, cur_b] - mean) / stdv
        thr = thresholds[i]
        out[i] = 1 if z > thr else (-1 if z < -thr else 0)
    return out


def full_recompute_corr(price_rb, widx, pairs, lookback):
    n_pairs = len(pairs)//2
    out = np.zeros(n_pairs, dtype=np.float64)
    n_symbols, window = price_rb.shape
    for i in range(n_pairs):
        a, b = pairs[2*i], pairs[2*i+1]
        wa, wb = widx[a], widx[b]
        sa = wa - lookback
        if sa < 0: sa += window
        sb = wb - lookback
        if sb < 0: sb += window
        ia, ib = sa, sb
        xa = np.zeros(lookback, dtype=np.float64)
        xb = np.zeros(lookback, dtype=np.float64)
        for k in range(lookback):
            xa[k] = price_rb[a, ia]
            xb[k] = price_rb[b, ib]
            ia += 1; ib += 1
            if ia == window: ia = 0
            if ib == window: ib = 0
        mx = xa.mean(); my = xb.mean()
        vx = xa.var(); vy = xb.var()
        cov = ((xa - mx)*(xb - my)).mean()
        out[i] = 0.0 if vx <= 0.0 or vy <= 0.0 else cov / np.sqrt(vx*vy)
    return out


class TestConsistencyIncrementalVsFull(unittest.TestCase):
    def test_consistency_random(self):
        rng = np.random.default_rng(42)
        n_symbols = 8
        window = 128
        lookback = 64
        n_pairs = 6
        # ring buffer data
        price_rb = rng.normal(100.0, 1.5, size=(n_symbols, window)).astype(np.float64)
        widx = rng.integers(0, window, size=n_symbols, dtype=np.int32)
        # generate random pairs
        syms = rng.choice(n_symbols, size=2*n_pairs, replace=True)
        pairs = syms.astype(np.int32)
        thresholds = np.full(n_pairs, 2.0, dtype=np.float64)
        # incremental calls need state arrays
        sums = np.zeros(n_pairs, dtype=np.float64)
        sumsq = np.zeros(n_pairs, dtype=np.float64)
        sx = np.zeros(n_pairs, dtype=np.float64)
        sy = np.zeros(n_pairs, dtype=np.float64)
        sxx = np.zeros(n_pairs, dtype=np.float64)
        syy = np.zeros(n_pairs, dtype=np.float64)
        sxy = np.zeros(n_pairs, dtype=np.float64)
        # initialize once
        sig_inc = zscore_batch_rb_inc(price_rb, widx, pairs, lookback, thresholds, sums, sumsq, 0)
        corr_inc = corr_pairs_rb_inc(price_rb, widx, pairs, lookback, sx, sy, sxx, syy, sxy, 0)
        # full recompute
        sig_full = full_recompute_zscores(price_rb, widx, pairs, lookback, thresholds)
        corr_full = full_recompute_corr(price_rb, widx, pairs, lookback)
        # compare
        np.testing.assert_array_equal(sig_inc, sig_full)
        np.testing.assert_allclose(corr_inc, corr_full, rtol=1e-12, atol=1e-9)


if __name__ == '__main__':
    unittest.main()

