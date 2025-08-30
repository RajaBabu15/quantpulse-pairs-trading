#include <hls_stream.h>
#include <ap_int.h>
#include <math.h>

extern "C" {

// Types: use double for parity with host; switch to fixed-point for throughput
void rolling_stats_kernel(
    const double* __restrict price_rb, // [n_symbols * window]
    const int*    __restrict write_idx,// [n_symbols]
    const int*    __restrict pairs,    // [2 * n_pairs]
    int n_symbols,
    int window,
    int lookback,
    // in/out rolling state
    double* __restrict sx,
    double* __restrict sy,
    double* __restrict sxx,
    double* __restrict syy,
    double* __restrict sxy,
    // outputs
    signed char* __restrict signals,
    const double* __restrict thresholds,
    double* __restrict correlations,
    // control
    int n_pairs,
    int initialized
) {
#pragma HLS INTERFACE m_axi port=price_rb   bundle=gmem0 depth=65536 offset=slave
#pragma HLS INTERFACE m_axi port=write_idx  bundle=gmem1 depth=4096  offset=slave
#pragma HLS INTERFACE m_axi port=pairs      bundle=gmem2 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=sx         bundle=gmem3 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=sy         bundle=gmem4 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=sxx        bundle=gmem5 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=syy        bundle=gmem6 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=sxy        bundle=gmem7 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=signals    bundle=gmem8 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=thresholds bundle=gmem9 depth=8192  offset=slave
#pragma HLS INTERFACE m_axi port=correlations bundle=gmem10 depth=8192 offset=slave
#pragma HLS INTERFACE s_axilite port=price_rb   bundle=control
#pragma HLS INTERFACE s_axilite port=write_idx  bundle=control
#pragma HLS INTERFACE s_axilite port=pairs      bundle=control
#pragma HLS INTERFACE s_axilite port=n_symbols  bundle=control
#pragma HLS INTERFACE s_axilite port=window     bundle=control
#pragma HLS INTERFACE s_axilite port=lookback   bundle=control
#pragma HLS INTERFACE s_axilite port=sx         bundle=control
#pragma HLS INTERFACE s_axilite port=sy         bundle=control
#pragma HLS INTERFACE s_axilite port=sxx        bundle=control
#pragma HLS INTERFACE s_axilite port=syy        bundle=control
#pragma HLS INTERFACE s_axilite port=sxy        bundle=control
#pragma HLS INTERFACE s_axilite port=signals    bundle=control
#pragma HLS INTERFACE s_axilite port=thresholds bundle=control
#pragma HLS INTERFACE s_axilite port=correlations bundle=control
#pragma HLS INTERFACE s_axilite port=n_pairs    bundle=control
#pragma HLS INTERFACE s_axilite port=initialized bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    PAIRS_LOOP: for (int i = 0; i < n_pairs; ++i) {
#pragma HLS PIPELINE II=1
        int a = pairs[2*i];
        int b = pairs[2*i + 1];
        int wa = write_idx[a];
        int wb = write_idx[b];
        if (!initialized) {
            int sa = wa - lookback; if (sa < 0) sa += window;
            int sb = wb - lookback; if (sb < 0) sb += window;
            double sxv=0.0, syv=0.0, sxxv=0.0, syyv=0.0, sxyv=0.0;
            int ia = sa, ib = sb;
            for (int k = 0; k < lookback; ++k) {
#pragma HLS UNROLL factor=2
                double xa = price_rb[(long)a*window + ia];
                double xb = price_rb[(long)b*window + ib];
                sxv += xa; syv += xb;
                sxxv += xa*xa; syyv += xb*xb; sxyv += xa*xb;
                ia++; if (ia == window) ia = 0;
                ib++; if (ib == window) ib = 0;
            }
            sx[i]=sxv; sy[i]=syv; sxx[i]=sxxv; syy[i]=syyv; sxy[i]=sxyv;
        } else {
            int newa = wa - 1; if (newa < 0) newa += window;
            int newb = wb - 1; if (newb < 0) newb += window;
            int olda = wa - lookback; if (olda < 0) olda += window;
            int oldb = wb - lookback; if (oldb < 0) oldb += window;
            double xa_new = price_rb[(long)a*window + newa];
            double xb_new = price_rb[(long)b*window + newb];
            double xa_old = price_rb[(long)a*window + olda];
            double xb_old = price_rb[(long)b*window + oldb];
            sx[i]  += (xa_new - xa_old);
            sy[i]  += (xb_new - xb_old);
            sxx[i] += (xa_new*xa_new - xa_old*xa_old);
            syy[i] += (xb_new*xb_new - xb_old*xb_old);
            sxy[i] += (xa_new*xb_new  - xa_old*xb_old);
        }
        double n = (double)lookback;
        double mx = sx[i]/n, my = sy[i]/n;
        double varx = sxx[i]/n - mx*mx;
        double vary = syy[i]/n - my*my;
        double cov  = sxy[i]/n - mx*my;
        double corr = (varx <= 0.0 || vary <= 0.0) ? 0.0 : (cov / sqrt(varx*vary));
        correlations[i] = corr;
        // z-score of spread
        int newa = wa - 1; if (newa < 0) newa += window;
        int newb = wb - 1; if (newb < 0) newb += window;
        int olda = wa - lookback; if (olda < 0) olda += window;
        int oldb = wb - lookback; if (oldb < 0) oldb += window;
        double s_new = price_rb[(long)a*window + newa] - price_rb[(long)b*window + newb];
        double s_old = price_rb[(long)a*window + olda] - price_rb[(long)b*window + oldb];
        static double s_sum[4096];
        static double s_sumsq[4096];
        double sum  = s_sum[i]   + (initialized ? (s_new - s_old) : 0.0);
        double sumsq= s_sumsq[i] + (initialized ? (s_new*s_new - s_old*s_old) : 0.0);
        if (!initialized) { // initialize with full window
            sum = 0.0; sumsq = 0.0;
            int sa = wa - lookback; if (sa < 0) sa += window;
            int sb = wb - lookback; if (sb < 0) sb += window;
            int ia = sa, ib = sb;
            for (int k = 0; k < lookback; ++k) {
                double sp = price_rb[(long)a*window + ia] - price_rb[(long)b*window + ib];
                sum += sp; sumsq += sp*sp;
                ia++; if (ia == window) ia = 0;
                ib++; if (ib == window) ib = 0;
            }
        }
        double mean = sum/n;
        double var = sumsq/n - mean*mean;
        signed char sig = 0;
        if (var > 0.0) {
            double stdv = sqrt(var);
            double z = (s_new - mean)/stdv;
            double thr = thresholds[i];
            sig = (z > thr) ? 1 : ((z < -thr) ? -1 : 0);
        }
        s_sum[i] = sum; s_sumsq[i] = sumsq;
        signals[i] = sig;
    }
}

} // extern "C"

