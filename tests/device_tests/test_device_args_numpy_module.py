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
@device(Host, numpy_module_arg='xp')
def create_array_init_5_at_host(shape, dtype, order, xp):
    return xp.full(shape, 5, dtype=dtype, order=order)


@device(CUDAGPU, numpy_module_arg='xp')
def create_array_init_5_at_gpu(shape, dtype, order, xp):
    return xp.full(shape, 5, dtype=dtype, order=order)


@device(VE, numpy_module_arg='xp')
def create_array_init_5_at_ve(shape, dtype, order, xp):
    return xp.full(shape, 5, dtype=dtype, order=order)


@pytest.mark.parametrize('shape', [(2), (2, 2), (2, 2, 2), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize('dtype', [
    'i4', 'i8', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16'
    ])
@pytest.mark.parametrize('order', ['C', 'F'])
class TestDeviceArgs:
    def test_device_args_host(self, shape, dtype, order):
        y = create_array_init_5_at_host(shape, dtype, order)
        assert(isinstance(y, np.ndarray))
        expected = np.full(shape, 5, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_cupy
    def test_device_args_gpu(self, shape, dtype, order):
        y = create_array_init_5_at_gpu(shape, dtype, order)
        assert(isinstance(y, cp.ndarray))
        expected = cp.full(shape, 5, dtype=dtype, order=order)
        assert((y == expected).all())

    @no_nlcpy
    def test_device_args_ve(self, shape, dtype, order):
        y = create_array_init_5_at_ve(shape, dtype, order)
        assert(isinstance(y, vp.ndarray))
        expected = vp.full(shape, 5, dtype=dtype, order=order)
        assert((y == expected).all())
