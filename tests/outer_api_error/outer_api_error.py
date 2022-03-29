import nlcpy
import cupy
import orchespy
from orchespy.devicetype import CUDAGPU, VE

import os

import pytest

no_ve_mem_cpy = pytest.mark.skipif(
        'liboverride_veo_memcpy.so' not in os.getenv('LD_PRELOAD'),
        reason='Not veo_read_mem failed')
no_cuda_mem_cpy = pytest.mark.skipif(
        'liboverride_cuda_memcpy.so' not in os.getenv('LD_PRELOAD'),
        reason='Not cuda_read_mem failed')


@no_ve_mem_cpy
def test_transfer_ve_memcpy_error():
    print('env is = ', os.getenv('LD_PRELOAD'))
    with pytest.raises(Exception) as e:
        vx = nlcpy.arange(10)
        orchespy.transfer_array(vx, CUDAGPU)

    assert str(e.value) == "veo_read_mem failed"


@no_cuda_mem_cpy
def test_transfer_cuda_memcpy_to_device_error():
    print('env is = ', os.getenv('LD_PRELOAD'))
    with pytest.raises(Exception) as e:
        vx = nlcpy.arange(10)
        orchespy.transfer_array(vx, CUDAGPU)

    assert str(e.value) == "cudaMemcpy failed"


@no_ve_mem_cpy
def test_transfer_ve_mem_write_error():
    print('env is = ', os.getenv('LD_PRELOAD'))
    with pytest.raises(Exception) as e:
        vx = cupy.arange(10)
        orchespy.transfer_array(vx, VE)

    assert str(e.value) == "veo_write_mem failed"


@no_cuda_mem_cpy
def test_transfer_cuda_memcpy_to_host_error():
    print('env is = ', os.getenv('LD_PRELOAD'))
    with pytest.raises(Exception) as e:
        vx = cupy.arange(10)
        orchespy.transfer_array(vx, VE)

    assert str(e.value) == "cudaMemcpy failed"
