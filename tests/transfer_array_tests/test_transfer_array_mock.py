import orchespy
from orchespy.devicetype import CUDAGPU
import sys
import pytest

# import numpy as np

if "nlcpy" in sys.modules:
    import nlcpy as vp

no_nlcpy = pytest.mark.skipif(
        "nlcpy" not in sys.modules, reason=' test require nlcpy. ')
no_cupy = pytest.mark.skipif(
        "cupy" not in sys.modules, reason=' test require cupy. ')


# Transfer 2d int zero array from VE to Host
@pytest.mark.parametrize('shape', [(2, 2)])
@pytest.mark.parametrize('dtype', [
    'i4'
    ])
@pytest.mark.parametrize('order', ['C'])
class TestTransfer:
    # Transfer zero array from VE to Host
    # Transfer one array from VE to GPU
    @no_cupy
    @no_nlcpy
    def test_transfer_one_v2g_on_host(self, shape, dtype, order, mocker):
        src = vp.ones(shape, dtype=dtype, order=order)
        mocker.patch(
                'orchespy.devicetype.ve.VE.find_device').return_value = None
        dst = orchespy.transfer_array(src, CUDAGPU)
        # check data type
        # check elements
        assert((dst == src).all())
