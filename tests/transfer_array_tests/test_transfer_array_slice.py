import orchespy
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


# Transfer 2d int zero array from VE to Host
@pytest.mark.parametrize('shape', [
    (10,), (10, 10), (10, 10, 10), (10, 20), (20, 10), (10, 20, 30)
    ])
@pytest.mark.parametrize('dtype', [
    'i4', 'i8', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16'
    ])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('sld', [
    (3, 7, None), (None, None, 2), (None, None, -1)
    ])
class TestTransfer:
    # Transfer zero array from VE to Host
    @no_nlcpy
    @no_cupy
    def test_transfer_slice_v2h(self, shape, dtype, order, sld):
        sl = slice(sld[0], sld[1], sld[2])
        num = 1
        for itm in shape:
            num = num * itm
        src_t = vp.arange(num, dtype=dtype).reshape(shape, order=order)
        src = src_t[sl]
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected_t = np.arange(num, dtype=dtype).reshape(shape, order=order)
        expected = expected_t[sl]
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from VE to GPU
    @no_nlcpy
    @no_cupy
    def test_transfer_slice_v2g(self, shape, dtype, order, sld):
        sl = slice(sld[0], sld[1], sld[2])
        num = 1
        for itm in shape:
            num = num * itm
        src_t = vp.arange(num, dtype=dtype).reshape(shape, order=order)
        src = src_t[sl]
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected_t = cp.arange(num, dtype=dtype).reshape(shape, order=order)
        expected = expected_t[sl]
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from Host to VE
    @no_nlcpy
    def test_transfer_slice_h2v(self, shape, dtype, order, sld):
        sl = slice(sld[0], sld[1], sld[2])
        num = 1
        for itm in shape:
            num = num * itm
        src_t = np.arange(num, dtype=dtype).reshape(shape, order=order)
        src = src_t[sl]
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected_t = vp.arange(num, dtype=dtype).reshape(shape, order=order)
        expected = expected_t[sl]
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from Host to GPU
    @no_cupy
    def test_transfer_slice_h2g(self, shape, dtype, order, sld):
        sl = slice(sld[0], sld[1], sld[2])
        num = 1
        for itm in shape:
            num = num * itm
        src_t = np.arange(num, dtype=dtype).reshape(shape, order=order)
        src = src_t[sl]
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected_t = cp.arange(num, dtype=dtype).reshape(shape, order=order)
        expected = expected_t[sl]
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from GPU to Host
    @no_cupy
    def test_transfer_slice_g2h(self, shape, dtype, order, sld):
        sl = slice(sld[0], sld[1], sld[2])
        num = 1
        for itm in shape:
            num = num * itm
        src_t = cp.arange(num, dtype=dtype).reshape(shape, order=order)
        src = src_t[sl]
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected_t = np.arange(num, dtype=dtype).reshape(shape, order=order)
        expected = expected_t[sl]
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from GPU to VE
    @no_nlcpy
    @no_cupy
    def test_transfer_slice_g2v(self, shape, dtype, order, sld):
        sl = slice(sld[0], sld[1], sld[2])
        num = 1
        for itm in shape:
            num = num * itm
        src_t = cp.arange(num, dtype=dtype).reshape(shape, order=order)
        src = src_t[sl]
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected_t = vp.arange(num, dtype=dtype).reshape(shape, order=order)
        expected = expected_t[sl]
        assert(src.strides == expected.strides)
        assert((dst == expected).all())
