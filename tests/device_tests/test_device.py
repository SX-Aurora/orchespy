from orchespy import device
from orchespy.devicetype import CUDAGPU, Host, VE
import sys
import pytest

import numpy as np

if "cupy" in sys.modules:
    import cupy as cp
if "nlcpy" in sys.modules:
    import nlcpy as vp

no_nlcpy = pytest.mark.skipif(
        "nlcpy" not in sys.modules, reason=' test require nlcpy. ')
no_cupy = pytest.mark.skipif(
        "cupy" not in sys.modules, reason=' test require cupy. ')


# for tests with an argument
@device(Host)
def id_host(x):
    return x


@device(CUDAGPU)
def id_gpu(x):
    return x


@device(VE)
def id_ve(x):
    return x


@pytest.mark.parametrize('shape', [(2), (2, 2), (2, 2, 2), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize('dtype', [
    'i4', 'i8', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16'
    ])
@pytest.mark.parametrize('order', ['C', 'F'])
class TestDevice:
    def test_device_one_arg_np_zeros_host(self, shape, dtype, order):
        x = np.zeros(shape, dtype=dtype, order=order)
        y = id_host(x)
        assert(y is x)

    @no_cupy
    def test_device_one_arg_np_zeros_gpu(self, shape, dtype, order):
        x = np.zeros(shape, dtype=dtype, order=order)
        y = id_gpu(x)
        assert(isinstance(y, cp.ndarray))
        expected = cp.zeros(shape, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_one_arg_np_zeros_ve(self, shape, dtype, order):
        x = np.zeros(shape, dtype=dtype, order=order)
        y = id_ve(x)
        assert(isinstance(y, vp.ndarray))
        expected = vp.zeros(shape, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_one_arg_cp_zeros_host(self, shape, dtype, order):
        x = cp.zeros(shape, dtype=dtype, order=order)
        y = id_host(x)
        assert(isinstance(y, np.ndarray))
        expected = np.zeros(shape, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_one_arg_cp_zeros_gpu(self, shape, dtype, order):
        x = cp.zeros(shape, dtype=dtype, order=order)
        y = id_gpu(x)
        assert(y is x)

    @no_cupy
    @no_nlcpy
    def test_device_one_arg_cp_zeros_ve(self, shape, dtype, order):
        x = cp.zeros(shape, dtype=dtype, order=order)
        y = id_ve(x)
        assert(isinstance(y, vp.ndarray))
        expected = vp.zeros(shape, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_one_arg_vp_zeros_host(self, shape, dtype, order):
        x = vp.zeros(shape, dtype=dtype, order=order)
        y = id_host(x)
        assert(isinstance(y, np.ndarray))
        expected = np.zeros(shape, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_one_arg_vp_zeros_gpu(self, shape, dtype, order):
        x = vp.zeros(shape, dtype=dtype, order=order)
        y = id_gpu(x)
        assert(isinstance(y, cp.ndarray))
        expected = cp.zeros(shape, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_one_arg_vp_zeros_ve(self, shape, dtype, order):
        x = vp.zeros(shape, dtype=dtype, order=order)
        y = id_ve(x)
        assert(y is x)
