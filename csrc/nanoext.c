#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include <sys/mman.h>

// Optional SIMD intrinsics
#if defined(__AVX2__) || defined(__x86_64__)
  #include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
#endif

// Utility: round up to next power of two
static inline uint64_t next_pow2_u64(uint64_t x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;
    return x;
}

// ==========================
// zscore_batch on ring buffer
// ==========================

static PyObject* zscore_batch_rb(PyObject* self, PyObject* args) {
    PyObject *price_rb_obj, *write_idx_obj, *pair_idx_obj, *thresholds_obj;
    int lookback;
    if (!PyArg_ParseTuple(args, "OOOiO", &price_rb_obj, &write_idx_obj, &pair_idx_obj, &lookback, &thresholds_obj)) {
        return NULL;
    }

    PyArrayObject* price_rb = (PyArrayObject*)PyArray_FROM_OTF(price_rb_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* write_idx = (PyArrayObject*)PyArray_FROM_OTF(write_idx_obj, NPY_INT32, NPY_ARRAY_CARRAY);
    PyArrayObject* pair_idx = (PyArrayObject*)PyArray_FROM_OTF(pair_idx_obj, NPY_INT32, NPY_ARRAY_CARRAY);
    PyArrayObject* thresholds = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);

    if (!price_rb || !write_idx || !pair_idx || !thresholds) {
        Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(thresholds);
        return NULL;
    }

    if (PyArray_NDIM(price_rb) != 2 || PyArray_NDIM(write_idx) != 1 || PyArray_NDIM(pair_idx) != 1 || PyArray_NDIM(thresholds) != 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid array dimensions");
        goto fail;
    }

    const npy_intp n_symbols = PyArray_DIM(price_rb, 0);
    const npy_intp window = PyArray_DIM(price_rb, 1);
    const npy_intp n_pairs_flat = PyArray_DIM(pair_idx, 0);
    if (n_pairs_flat % 2 != 0) {
        PyErr_SetString(PyExc_ValueError, "pair_indices length must be even");
        goto fail;
    }
    const npy_intp n_pairs = n_pairs_flat / 2;
    if (PyArray_DIM(thresholds, 0) != n_pairs) {
        PyErr_SetString(PyExc_ValueError, "thresholds length mismatch");
        goto fail;
    }
    if (lookback <= 0 || lookback > (int)window) {
        PyErr_SetString(PyExc_ValueError, "lookback must be in (0, window]");
        goto fail;
    }

    double* rb = (double*)PyArray_DATA(price_rb);
    int32_t* widx = (int32_t*)PyArray_DATA(write_idx);
    int32_t* pairs = (int32_t*)PyArray_DATA(pair_idx);
    double* th = (double*)PyArray_DATA(thresholds);

    npy_intp out_dims[1] = {n_pairs};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, out_dims, NPY_INT8);
    if (!out) goto fail;
    int8_t* out_data = (int8_t*)PyArray_DATA(out);

    // Compute signals per pair
    for (npy_intp i = 0; i < n_pairs; ++i) {
        int32_t idx1 = pairs[2*i];
        int32_t idx2 = pairs[2*i + 1];
        double thr = th[i];
        if (idx1 < 0 || idx2 < 0 || idx1 >= n_symbols || idx2 >= n_symbols) {
            out_data[i] = 0;
            continue;
        }
        int32_t w1 = widx[idx1];
        int32_t w2 = widx[idx2];

        // Rolling window start indices
        int32_t start1 = w1 - lookback;
        if (start1 < 0) start1 += (int32_t)window;
        int32_t start2 = w2 - lookback;
        if (start2 < 0) start2 += (int32_t)window;

        // Compute mean and variance of spread over lookback
        double sum = 0.0, sumsq = 0.0;
        int32_t idx_1 = start1;
        int32_t idx_2 = start2;
        for (int k = 0; k < lookback; ++k) {
            double p1 = rb[(size_t)idx1 * window + (size_t)idx_1];
            double p2 = rb[(size_t)idx2 * window + (size_t)idx_2];
            double s = p1 - p2;
            sum += s;
            sumsq += s * s;
            idx_1++; if (idx_1 == window) idx_1 = 0;
            idx_2++; if (idx_2 == window) idx_2 = 0;
        }
        double mean = sum / (double)lookback;
        double var = (sumsq / (double)lookback) - mean * mean;
        if (var <= 0.0) { out_data[i] = 0; continue; }
        double std = sqrt(var);

        // Current spread = last elements (position just before w1/w2)
        int32_t cur1 = w1 - 1; if (cur1 < 0) cur1 += (int32_t)window;
        int32_t cur2 = w2 - 1; if (cur2 < 0) cur2 += (int32_t)window;
        double current_spread = rb[(size_t)idx1 * window + (size_t)cur1] - rb[(size_t)idx2 * window + (size_t)cur2];
        double z = (current_spread - mean) / std;
        if (z > thr) out_data[i] = 1; else if (z < -thr) out_data[i] = -1; else out_data[i] = 0;
    }

    Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(thresholds);
    return (PyObject*)out;

fail:
    Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(thresholds);
    return NULL;
}

// ==========================
// Incremental zscore on ring buffer (O(n_pairs))
// ==========================

// Memory advice: set MADV_SEQUENTIAL (best-effort) and prefault via touching pages
static PyObject* mem_advise_sequential(PyObject* self, PyObject* args) {
    PyObject* arr_obj;
    if (!PyArg_ParseTuple(args, "O", &arr_obj)) return NULL;
    PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(arr_obj, NPY_NOTYPE, NPY_ARRAY_CARRAY);
    if (!arr) return NULL;
    void* ptr = PyArray_DATA(arr);
    size_t len = (size_t)PyArray_NBYTES(arr);
    int rc = madvise(ptr, len, MADV_SEQUENTIAL);
    Py_DECREF(arr);
    return PyLong_FromLong((long)rc);
}

static PyObject* mem_prefault(PyObject* self, PyObject* args) {
    PyObject* arr_obj;
    if (!PyArg_ParseTuple(args, "O", &arr_obj)) return NULL;
    PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(arr_obj, NPY_NOTYPE, NPY_ARRAY_CARRAY);
    if (!arr) return NULL;
    volatile char* p = (volatile char*)PyArray_DATA(arr);
    size_t len = (size_t)PyArray_NBYTES(arr);
    // Touch one byte per 4KB page
    size_t page = 4096;
    for (size_t i = 0; i < len; i += page) {
        (void)p[i];
    }
    // Also touch the last byte
    if (len > 0) (void)p[len - 1];
    Py_DECREF(arr);
    Py_RETURN_NONE;
}

static PyObject* zscore_batch_rb_inc(PyObject* self, PyObject* args) {
    PyObject *price_rb_obj, *write_idx_obj, *pair_idx_obj, *thresholds_obj, *sums_obj, *sumsq_obj;
    int lookback;
    int initialized = 0;
    if (!PyArg_ParseTuple(args, "OOOiOOOi", &price_rb_obj, &write_idx_obj, &pair_idx_obj, &lookback, &thresholds_obj, &sums_obj, &sumsq_obj, &initialized)) {
        return NULL;
    }

    PyArrayObject* price_rb = (PyArrayObject*)PyArray_FROM_OTF(price_rb_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* write_idx = (PyArrayObject*)PyArray_FROM_OTF(write_idx_obj, NPY_INT32, NPY_ARRAY_CARRAY);
    PyArrayObject* pair_idx = (PyArrayObject*)PyArray_FROM_OTF(pair_idx_obj, NPY_INT32, NPY_ARRAY_CARRAY);
    PyArrayObject* thresholds = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* sums = (PyArrayObject*)PyArray_FROM_OTF(sums_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* sumsq = (PyArrayObject*)PyArray_FROM_OTF(sumsq_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);

    if (!price_rb || !write_idx || !pair_idx || !thresholds || !sums || !sumsq) {
        Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(thresholds);
        Py_XDECREF(sums); Py_XDECREF(sumsq); return NULL;
    }

    if (PyArray_NDIM(price_rb) != 2) { PyErr_SetString(PyExc_ValueError, "price_rb must be 2D"); goto fail_inc; }
    if (PyArray_NDIM(write_idx) != 1 || PyArray_NDIM(pair_idx) != 1 || PyArray_NDIM(thresholds) != 1 || PyArray_NDIM(sums) != 1 || PyArray_NDIM(sumsq) != 1) {
        PyErr_SetString(PyExc_ValueError, "index/threshold/state arrays have wrong dims");
        goto fail_inc;
    }

    const npy_intp n_symbols = PyArray_DIM(price_rb, 0);
    const npy_intp window = PyArray_DIM(price_rb, 1);
    const npy_intp n_pairs_flat = PyArray_DIM(pair_idx, 0);
    if (n_pairs_flat % 2 != 0) { PyErr_SetString(PyExc_ValueError, "pair_indices length must be even"); goto fail_inc; }
    const npy_intp n_pairs = n_pairs_flat / 2;
    if (PyArray_DIM(thresholds, 0) != n_pairs || PyArray_DIM(sums, 0) != n_pairs || PyArray_DIM(sumsq, 0) != n_pairs) {
        PyErr_SetString(PyExc_ValueError, "thresholds/sums length mismatch with pairs");
        goto fail_inc;
    }
    if (lookback <= 0 || lookback > (int)window) { PyErr_SetString(PyExc_ValueError, "invalid lookback"); goto fail_inc; }

    double* rb = (double*)PyArray_DATA(price_rb);
    int32_t* widx = (int32_t*)PyArray_DATA(write_idx);
    int32_t* pairs = (int32_t*)PyArray_DATA(pair_idx);
    double* th = (double*)PyArray_DATA(thresholds);
    double* sums_data = (double*)PyArray_DATA(sums);
    double* sumsq_data = (double*)PyArray_DATA(sumsq);

    npy_intp out_dims[1] = {n_pairs};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, out_dims, NPY_INT8);
    if (!out) goto fail_inc;
    int8_t* out_data = (int8_t*)PyArray_DATA(out);

    for (npy_intp i = 0; i < n_pairs; ++i) {
        int32_t idx1 = pairs[2*i];
        int32_t idx2 = pairs[2*i + 1];
        if (idx1 < 0 || idx2 < 0 || idx1 >= n_symbols || idx2 >= n_symbols) { out_data[i] = 0; continue; }
        int32_t w1 = widx[idx1];
        int32_t w2 = widx[idx2];

        if (!initialized) {
            // Compute initial sums across lookback
            int32_t s1 = w1 - lookback; if (s1 < 0) s1 += (int32_t)window;
            int32_t s2 = w2 - lookback; if (s2 < 0) s2 += (int32_t)window;
            double sum = 0.0, ssq = 0.0;
            int32_t j1 = s1, j2 = s2;
            for (int k = 0; k < lookback; ++k) {
                double p1 = rb[(size_t)idx1 * window + (size_t)j1];
                double p2 = rb[(size_t)idx2 * window + (size_t)j2];
                double s = p1 - p2;
                sum += s;
                ssq += s * s;
                j1++; if (j1 == window) j1 = 0;
                j2++; if (j2 == window) j2 = 0;
            }
            sums_data[i] = sum;
            sumsq_data[i] = ssq;
        } else {
            // Incremental update: add new, remove old
            int32_t new1 = w1 - 1; if (new1 < 0) new1 += (int32_t)window;
            int32_t new2 = w2 - 1; if (new2 < 0) new2 += (int32_t)window;
            int32_t old1 = w1 - lookback; if (old1 < 0) old1 += (int32_t)window;
            int32_t old2 = w2 - lookback; if (old2 < 0) old2 += (int32_t)window;
            // Prefetch likely next positions
#if defined(__GNUC__)
            __builtin_prefetch(&rb[(size_t)idx1 * window + (size_t)((new1 + 2) % (int32_t)window)], 0 /*read*/, 0 /*no locality*/);
            __builtin_prefetch(&rb[(size_t)idx2 * window + (size_t)((new2 + 2) % (int32_t)window)], 0, 0);
            __builtin_prefetch(&rb[(size_t)idx1 * window + (size_t)((new1 + 3) % (int32_t)window)], 0, 0);
            __builtin_prefetch(&rb[(size_t)idx2 * window + (size_t)((new2 + 3) % (int32_t)window)], 0, 0);
            #endif
            double s_new = rb[(size_t)idx1 * window + (size_t)new1] - rb[(size_t)idx2 * window + (size_t)new2];
            double s_old = rb[(size_t)idx1 * window + (size_t)old1] - rb[(size_t)idx2 * window + (size_t)old2];
            sums_data[i] += (s_new - s_old);
            sumsq_data[i] += (s_new*s_new - s_old*s_old);
        }

        double mean = sums_data[i] / (double)lookback;
        double var = (sumsq_data[i] / (double)lookback) - mean * mean;
        if (var <= 0.0) { out_data[i] = 0; continue; }
        double stdv = sqrt(var);
        int32_t cur1 = w1 - 1; if (cur1 < 0) cur1 += (int32_t)window;
        int32_t cur2 = w2 - 1; if (cur2 < 0) cur2 += (int32_t)window;
        double current_spread = rb[(size_t)idx1 * window + (size_t)cur1] - rb[(size_t)idx2 * window + (size_t)cur2];
        double z = (current_spread - mean) / stdv;
        double thr = th[i];
        out_data[i] = (z > thr) ? 1 : ((z < -thr) ? -1 : 0);
    }

    Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(thresholds); Py_XDECREF(sums); Py_XDECREF(sumsq);
    return (PyObject*)out;

fail_inc:
    Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(thresholds); Py_XDECREF(sums); Py_XDECREF(sumsq);
    return NULL;
}

// ==========================
// Incremental pairwise correlation on ring buffer (rolling updates)
static PyObject* corr_pairs_rb_inc(PyObject* self, PyObject* args) {
    PyObject *price_rb_obj, *write_idx_obj, *pair_idx_obj, *sx_obj, *sy_obj, *sxx_obj, *syy_obj, *sxy_obj;
    int lookback, initialized = 0;
    if (!PyArg_ParseTuple(args, "OOOiOOOOOi", &price_rb_obj, &write_idx_obj, &pair_idx_obj, &lookback, &sx_obj, &sy_obj, &sxx_obj, &syy_obj, &sxy_obj, &initialized)) {
        return NULL;
    }
    PyArrayObject* price_rb = (PyArrayObject*)PyArray_FROM_OTF(price_rb_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* write_idx = (PyArrayObject*)PyArray_FROM_OTF(write_idx_obj, NPY_INT32, NPY_ARRAY_CARRAY);
    PyArrayObject* pair_idx = (PyArrayObject*)PyArray_FROM_OTF(pair_idx_obj, NPY_INT32, NPY_ARRAY_CARRAY);
    PyArrayObject* sx = (PyArrayObject*)PyArray_FROM_OTF(sx_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* sy = (PyArrayObject*)PyArray_FROM_OTF(sy_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* sxx = (PyArrayObject*)PyArray_FROM_OTF(sxx_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* syy = (PyArrayObject*)PyArray_FROM_OTF(syy_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    PyArrayObject* sxy = (PyArrayObject*)PyArray_FROM_OTF(sxy_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    if (!price_rb || !write_idx || !pair_idx || !sx || !sy || !sxx || !syy || !sxy) {
        Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx);
        Py_XDECREF(sx); Py_XDECREF(sy); Py_XDECREF(sxx); Py_XDECREF(syy); Py_XDECREF(sxy);
        return NULL;
    }
    const npy_intp n_symbols = PyArray_DIM(price_rb, 0);
    const npy_intp window = PyArray_DIM(price_rb, 1);
    const npy_intp n_pairs_flat = PyArray_DIM(pair_idx, 0);
    if (n_pairs_flat % 2 != 0) { PyErr_SetString(PyExc_ValueError, "pair_indices length must be even"); goto fail_cpi; }
    const npy_intp n_pairs = n_pairs_flat / 2;
    if (PyArray_DIM(sx, 0) != n_pairs || PyArray_DIM(sy, 0) != n_pairs || PyArray_DIM(sxx, 0) != n_pairs || PyArray_DIM(syy, 0) != n_pairs || PyArray_DIM(sxy, 0) != n_pairs) {
        PyErr_SetString(PyExc_ValueError, "state arrays length mismatch");
        goto fail_cpi;
    }
    if (lookback <= 0 || lookback > (int)window) { PyErr_SetString(PyExc_ValueError, "invalid lookback"); goto fail_cpi; }

    double* rb = (double*)PyArray_DATA(price_rb);
    int32_t* widx = (int32_t*)PyArray_DATA(write_idx);
    int32_t* pairs = (int32_t*)PyArray_DATA(pair_idx);
    double* psx = (double*)PyArray_DATA(sx);
    double* psy = (double*)PyArray_DATA(sy);
    double* psxx = (double*)PyArray_DATA(sxx);
    double* psyy = (double*)PyArray_DATA(syy);
    double* psxy = (double*)PyArray_DATA(sxy);

    npy_intp out_dims[1] = {n_pairs};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    if (!out) goto fail_cpi;
    double* out_corr = (double*)PyArray_DATA(out);

    for (npy_intp i = 0; i < n_pairs; ++i) {
        int32_t a = pairs[2*i];
        int32_t b = pairs[2*i + 1];
        if (a < 0 || b < 0 || a >= n_symbols || b >= n_symbols) { out_corr[i] = 0.0; continue; }
        int32_t wa = widx[a];
        int32_t wb = widx[b];

        if (!initialized) {
            // Initialize rolling sums
            int32_t sa = wa - lookback; if (sa < 0) sa += (int32_t)window;
            int32_t sb = wb - lookback; if (sb < 0) sb += (int32_t)window;
            double sxv=0.0, syv=0.0, sxxv=0.0, syyv=0.0, sxyv=0.0;
            int32_t ia = sa, ib = sb;
            for (int k = 0; k < lookback; ++k) {
                double xa = rb[(size_t)a * window + (size_t)ia];
                double xb = rb[(size_t)b * window + (size_t)ib];
                sxv += xa; syv += xb;
                sxxv += xa*xa; syyv += xb*xb; sxyv += xa*xb;
                ia++; if (ia == window) ia = 0;
                ib++; if (ib == window) ib = 0;
            }
            psx[i] = sxv; psy[i] = syv; psxx[i] = sxxv; psyy[i] = syyv; psxy[i] = sxyv;
        } else {
            // Incremental update
            int32_t newa = wa - 1; if (newa < 0) newa += (int32_t)window;
            int32_t newb = wb - 1; if (newb < 0) newb += (int32_t)window;
            int32_t olda = wa - lookback; if (olda < 0) olda += (int32_t)window;
            int32_t oldb = wb - lookback; if (oldb < 0) oldb += (int32_t)window;
#if defined(__GNUC__)
            __builtin_prefetch(&rb[(size_t)a * window + (size_t)((newa + 2) % (int32_t)window)], 0, 0);
            __builtin_prefetch(&rb[(size_t)b * window + (size_t)((newb + 2) % (int32_t)window)], 0, 0);
            __builtin_prefetch(&rb[(size_t)a * window + (size_t)((newa + 3) % (int32_t)window)], 0, 0);
            __builtin_prefetch(&rb[(size_t)b * window + (size_t)((newb + 3) % (int32_t)window)], 0, 0);
            #endif
            double xa_new = rb[(size_t)a * window + (size_t)newa];
            double xb_new = rb[(size_t)b * window + (size_t)newb];
            double xa_old = rb[(size_t)a * window + (size_t)olda];
            double xb_old = rb[(size_t)b * window + (size_t)oldb];
            psx[i] += (xa_new - xa_old);
            psy[i] += (xb_new - xb_old);
            psxx[i] += (xa_new*xa_new - xa_old*xa_old);
            psyy[i] += (xb_new*xb_new - xb_old*xb_old);
            psxy[i] += (xa_new*xb_new - xa_old*xb_old);
        }
        double n = (double)lookback;
        double mx = psx[i] / n;
        double my = psy[i] / n;
        double varx = psxx[i] / n - mx*mx;
        double vary = psyy[i] / n - my*my;
        double cov = psxy[i] / n - mx*my;
        double corr = (varx <= 0.0 || vary <= 0.0) ? 0.0 : (cov / sqrt(varx*vary));
        out_corr[i] = corr;
    }

    Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(sx); Py_XDECREF(sy); Py_XDECREF(sxx); Py_XDECREF(syy); Py_XDECREF(sxy);
    return (PyObject*)out;

fail_cpi:
    Py_XDECREF(price_rb); Py_XDECREF(write_idx); Py_XDECREF(pair_idx); Py_XDECREF(sx); Py_XDECREF(sy); Py_XDECREF(sxx); Py_XDECREF(syy); Py_XDECREF(sxy);
    return NULL;
}

static PyMethodDef module_methods[] = {
    {"zscore_batch_rb_inc", (PyCFunction)zscore_batch_rb_inc, METH_VARARGS, "Incremental Z-score batch on ring buffer with state: (price_rb, write_idx, pair_indices, lookback, thresholds, sums, sumsq, initialized) -> int8[n_pairs]"},
    {"corr_pairs_rb_inc", (PyCFunction)corr_pairs_rb_inc, METH_VARARGS, "Incremental correlation for pairs on ring buffer: (price_rb, write_idx, pair_indices, lookback, sx, sy, sxx, syy, sxy, initialized) -> float64[n_pairs]"},
    {"mem_advise_sequential", (PyCFunction)mem_advise_sequential, METH_VARARGS, "Best-effort madvise(MADV_SEQUENTIAL) on array memory; returns rc"},
    {"mem_prefault", (PyCFunction)mem_prefault, METH_VARARGS, "Touch pages of array to prefault into RAM"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nanoextmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "nanoext",
    .m_doc = "Nanosecond HFT native extensions: lock-free SPSC queue and ring-buffer zscore",
    .m_size = -1,
    .m_methods = module_methods
};

PyMODINIT_FUNC PyInit_nanoext(void) {
    import_array();
    PyObject* m = PyModule_Create(&nanoextmodule);
    if (!m) return NULL;
    return m;
}

