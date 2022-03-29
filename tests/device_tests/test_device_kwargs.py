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


# for tests with kwargs
@device(Host)
def matrix_addition_at_host(**kwargs):
    return kwargs.get('key1') + kwargs.get('key2')


@device(CUDAGPU)
def matrix_addition_at_gpu(**kwargs):
    return kwargs.get('key1') + kwargs.get('key2')


@device(VE)
def matrix_addition_at_ve(**kwargs):
    return kwargs.get('key1') + kwargs.get('key2')


@pytest.mark.parametrize('shape', [(2), (2, 2), (2, 2, 2), (2, 4), (2, 3, 4)])
@pytest.mark.parametrize('dtype', [
    'i4', 'i8', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16'
    ])
@pytest.mark.parametrize('order', ['C', 'F'])
class TestDeviceKwargs:
    def test_device_kwargs_np_host(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = np.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_host(key1=x1, key2=x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_kwargs_cp_host(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_host(key1=x1, key2=x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_kwargs_vp_host(self, shape, dtype, order):
        x1 = vp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_host(key1=x1, key2=x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_kwargs_np_cp_host(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_host(key1=x1, key2=x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_kwargs_np_vp_host(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_host(key1=x1, key2=x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_kwargs_cp_vp_host(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_host(key1=x1, key2=x2)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_kwargs_np_gpu(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = np.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_gpu(key1=x1, key2=x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_kwargs_cp_gpu(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_gpu(key1=x1, key2=x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_kwargs_vp_gpu(self, shape, dtype, order):
        x1 = vp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_gpu(key1=x1, key2=x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_kwargs_np_cp_gpu(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_gpu(key1=x1, key2=x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_kwargs_np_vp_gpu(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_gpu(key1=x1, key2=x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_kwargs_cp_vp_gpu(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_gpu(key1=x1, key2=x2)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_kwargs_np_ve(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = np.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_ve(key1=x1, key2=x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_kwargs_cp_ve(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_ve(key1=x1, key2=x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_kwargs_vp_ve(self, shape, dtype, order):
        x1 = vp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_ve(key1=x1, key2=x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_kwargs_np_cp_ve(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = cp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_ve(key1=x1, key2=x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_kwargs_np_vp_ve(self, shape, dtype, order):
        x1 = np.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_ve(key1=x1, key2=x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    @no_nlcpy
    def test_device_kwargs_cp_vp_ve(self, shape, dtype, order):
        x1 = cp.ones(shape, dtype=dtype, order=order)
        x2 = vp.ones(shape, dtype=dtype, order=order)
        y = matrix_addition_at_ve(key1=x1, key2=x2)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 2, dtype=dtype, order=order)
        assert((y == expected).all())
