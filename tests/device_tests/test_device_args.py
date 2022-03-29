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
def sum_at_host(*args):
    return sum(args)


@device(CUDAGPU)
def sum_at_gpu(*args):
    return sum(args)


@device(VE)
def sum_at_ve(*args):
    return sum(args)


@pytest.mark.parametrize('shape', [(2), (2, 2), (2, 2, 2), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize('dtype', [
    'i4', 'i8', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16'
    ])
@pytest.mark.parametrize('order', ['C', 'F'])
class TestDeviceArgs:
    def test_device_args_np_host(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = np.ones(shape, dtype=dtype, order=order)
        y = sum_at_host(x1, x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_args_cp_host(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = sum_at_host(x1, x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_args_vp_host(self, shape, dtype, order):
        x1 = vp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_host(x1, x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_args_np_cp_host(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = sum_at_host(x1, x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_args_np_vp_host(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_host(x1, x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_args_cp_vp_host(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_host(x1, x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_args_np_gpu(self, shape, dtype, order):
        print(shape, dtype, order)
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = np.ones(shape, dtype=dtype, order=order)
        y = sum_at_gpu(x1, x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_args_cp_gpu(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = sum_at_gpu(x1, x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_args_vp_gpu(self, shape, dtype, order):
        x1 = vp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_gpu(x1, x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_args_np_cp_gpu(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = sum_at_gpu(x1, x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_args_np_vp_gpu(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_gpu(x1, x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_args_cp_vp_gpu(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_gpu(x1, x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_args_np_ve(self, shape, dtype, order):
        print(shape, dtype, order)
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = np.ones(shape, dtype=dtype, order=order)
        y = sum_at_ve(x1, x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_args_cp_ve(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = sum_at_ve(x1, x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_args_vp_ve(self, shape, dtype, order):
        x1 = vp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_ve(x1, x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_args_np_cp_ve(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = sum_at_ve(x1, x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_args_np_vp_ve(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_ve(x1, x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_args_cp_vp_ve(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = sum_at_ve(x1, x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())
