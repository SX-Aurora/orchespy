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
@pytest.mark.parametrize('shape', [(1000, 1000)])
@pytest.mark.parametrize('dtype', [
    'i4', 'i8', 'f4', 'f8', 'c8', 'c16'
    ])
@pytest.mark.parametrize('order', ['C'])
class TestTransferHuge:
    # Transfer zero array from VE to Host
    @no_nlcpy
    def test_transfer_zero_v2h(self, shape, dtype, order):
        src = vp.zeros(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected = np.zeros(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from VE to GPU
    @no_nlcpy
    @no_cupy
    def test_transfer_zero_v2g(self, shape, dtype, order):
        src = vp.zeros(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected = cp.zeros(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from Host to VE
    @no_nlcpy
    def test_transfer_zero_h2v(self, shape, dtype, order):
        src = np.zeros(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected = vp.zeros(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from Host to GPU
    @no_cupy
    def test_transfer_zero_h2g(self, shape, dtype, order):
        src = np.zeros(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected = cp.zeros(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from GPU to Host
    @no_cupy
    def test_transfer_zero_g2h(self, shape, dtype, order):
        src = cp.zeros(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected = np.zeros(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer zero array from GPU to VE
    @no_nlcpy
    @no_cupy
    def test_transfer_zero_g2v(self, shape, dtype, order):
        src = cp.zeros(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected = vp.zeros(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer one array from VE to Host
    @no_nlcpy
    def test_transfer_one_v2h(self, shape, dtype, order):
        src = vp.ones(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected = np.ones(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer one array from VE to GPU
    @no_nlcpy
    @no_cupy
    def test_transfer_one_v2g(self, shape, dtype, order):
        src = vp.ones(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected = cp.ones(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer one array from Host to VE
    @no_nlcpy
    def test_transfer_one_h2v(self, shape, dtype, order):
        src = np.ones(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected = vp.ones(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer one array from Host to GPU
    @no_cupy
    def test_transfer_one_h2g(self, shape, dtype, order):
        src = np.ones(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected = cp.ones(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer one array from GPU to Host
    @no_cupy
    def test_transfer_one_g2h(self, shape, dtype, order):
        src = cp.ones(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected = np.ones(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer one array from GPU to VE
    @no_nlcpy
    @no_cupy
    def test_transfer_one_g2v(self, shape, dtype, order):
        src = cp.ones(shape, dtype=dtype, order=order)
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected = vp.ones(shape, dtype=dtype, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer any array from VE to Host
    @no_nlcpy
    def test_transfer_any_v2h(self, shape, dtype, order):
        num = 1
        for itm in shape:
            num = num * itm
        src_t = vp.linspace(1, num, num, dtype=dtype)
        src = src_t.reshape(shape, order=order)
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected_t = np.linspace(1, num, num, dtype=dtype)
        expected = expected_t.reshape(shape, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer any array from VE to GPU
    @no_nlcpy
    @no_cupy
    def test_transfer_any_v2g(self, shape, dtype, order):
        num = 1
        for itm in shape:
            num = num * itm
        src_t = vp.linspace(1, num, num, dtype=dtype)
        src = src_t.reshape(shape, order=order)
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected_t = cp.linspace(1, num, num, dtype=dtype)
        expected = expected_t.reshape(shape, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer any array from Host to VE
    @no_nlcpy
    def test_transfer_any_h2v(self, shape, dtype, order):
        num = 1
        for itm in shape:
            num = num * itm
        src_t = np.linspace(1, num, num, dtype=dtype)
        src = src_t.reshape(shape, order=order)
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected_t = vp.linspace(1, num, num, dtype=dtype)
        expected = expected_t.reshape(shape, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer any array from Host to GPU
    @no_cupy
    def test_transfer_any_h2g(self, shape, dtype, order):
        num = 1
        for itm in shape:
            num = num * itm
        src_t = np.linspace(1, num, num, dtype=dtype)
        src = src_t.reshape(shape, order=order)
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected_t = cp.linspace(1, num, num, dtype=dtype)
        expected = expected_t.reshape(shape, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer any array from GPU to Host
    @no_cupy
    def test_transfer_any_g2h(self, shape, dtype, order):
        num = 1
        for itm in shape:
            num = num * itm
        src_t = cp.linspace(1, num, num, dtype=dtype)
        src = src_t.reshape(shape, order=order)
        dst = orchespy.transfer_array(src, Host)
        # check data type
        assert(type(dst) is np.ndarray)
        # check elements
        expected_t = np.linspace(1, num, num, dtype=dtype)
        expected = expected_t.reshape(shape, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())

    # Transfer any array from GPU to VE
    @no_nlcpy
    @no_cupy
    def test_transfer_any_g2v(self, shape, dtype, order):
        num = 1
        for itm in shape:
            num = num * itm
        src_t = cp.linspace(1, num, num, dtype=dtype)
        src = src_t.reshape(shape, order=order)
        dst = orchespy.transfer_array(src, VE)
        # check data type
        assert(type(dst) is vp.ndarray)
        # check elements
        expected_t = vp.linspace(1, num, num, dtype=dtype)
        expected = expected_t.reshape(shape, order=order)
        assert(src.strides == expected.strides)
        assert((dst == expected).all())
