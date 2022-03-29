import orchespy
from orchespy.devicetype import CUDAGPU
import sys
import pytest


if "cupy" in sys.modules:
    import cupy as cp
if "nlcpy" in sys.modules:
    import nlcpy as vp
    from nlcpy.core import set_boundary_size

no_nlcpy = pytest.mark.skipif(
        "nlcpy" not in sys.modules, reason=' test require nlcpy. ')
no_cupy = pytest.mark.skipif(
        "cupy" not in sys.modules, reason=' test require cupy. ')


# Transfer 2d int zero array from VE to Host
@pytest.mark.parametrize('shape', [(2,), (2, 2), (2, 2, 2)])
@pytest.mark.parametrize('dtype', [
    'i4', 'i8', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16'
    ])
@pytest.mark.parametrize('order', ['C', 'F'])
class TestTransfer:
    # Transfer zero array from VE to Host
    # Transfer one array from VE to GPU
    @no_nlcpy
    @no_cupy
    def test_transfer_one_v2g_on_host(self, shape, dtype, order):
        set_boundary_size(40)
        src = vp.empty(shape, dtype=dtype, order=order)
        set_boundary_size(0)
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        assert(type(dst) is cp.ndarray)
        # check elements
        expected = cp.asarray(src.get())
        assert((dst == expected).all())
