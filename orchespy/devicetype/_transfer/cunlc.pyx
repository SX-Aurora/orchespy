# distutils: language = c++
# Translation module between CuPy on NVIDIA GPU and NLCPy on NEC Vector Engine
import nlcpy
import nlcpy.core
import nlcpy.core.core

from nlcpy.veo import _get_veo_proc

import cupy

from libc.stdlib cimport *
from libc.stdint cimport *

from nlcpy.veo._veo cimport *

import os

# TODO: remove when using native transfer library
cdef extern from '<cuda_runtime.h>' nogil:
    cdef enum cudaError:
        cudaSuccess
    ctypedef cudaError cudaError_t
    cdef enum cudaMemcpyKind:
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice

    cdef cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind) nogil

# TODO: use native transfer library
cdef _copy_from_ve_to_gpu_via_host(long dst_addr, long src_addr, size_t size):
    cdef VeoProc proc = _get_veo_proc()
    cdef void *buf = malloc(size)
    cdef int rv
    try:
        # VE -> Host
        rv = veo_read_mem(<veo_proc_handle *>proc.proc_handle,
                          buf, src_addr, size)
        if rv != 0:
            raise RuntimeError("veo_read_mem failed")
        # Host -> GPU
        rv = cudaMemcpy(<void *>dst_addr, <void *>buf, size,
                        cudaMemcpyHostToDevice)
        if rv != cudaSuccess:
            raise RuntimeError("cudaMemcpy failed")
    finally:
        free(buf)

# TODO: use native transfer library
cdef _copy_from_gpu_to_ve_via_host(long dst_addr, long src_addr, size_t size):
    cdef VeoProc proc = _get_veo_proc()
    cdef void *buf = malloc(size)
    cdef int rv
    try:
        # GPU -> Host
        rv = cudaMemcpy(<void *>buf, <void *>src_addr, size,
                        cudaMemcpyDeviceToHost)
        if rv != cudaSuccess:
            raise RuntimeError("cudaMemcpy failed")
        rv = veo_write_mem(<veo_proc_handle *>proc.proc_handle,
                           dst_addr, buf, size)
        if rv != 0:
            raise RuntimeError("veo_write_mem failed")
        # Host -> VE
    finally:
        free(buf)

cdef _get_func_copy_from_gpu_to_ve(data):
    # TODO: implement direct transfer and switch
    return _copy_from_gpu_to_ve_via_host

def convert_from_cupy_to_nlcpy(data):
    assert(isinstance(data, cupy.ndarray))
    if not data._c_contiguous and data._f_contiguous:
        order = 'F'
    else:
        order = 'C'

    copyfunc = _get_func_copy_from_gpu_to_ve(data)

    # assume nlcpy.dtype == cupy.dtype
    rv = nlcpy.empty(data.shape, data.dtype, order)
    # Flush all requests for ndarray rv to be allocated
    nlcpy.request.flush()

    cdef vemem = rv.ve_adr
    # transfer data to rv
    if data._c_contiguous or data._f_contiguous:
        copyfunc(vemem, data.data.ptr, data.nbytes)
    else:
        # Unless data is contiguous, create a temporary C-contiguous ndarray.
        tmp = cupy.ascontiguousarray(data)
        copyfunc(vemem, tmp.data.ptr, tmp.nbytes)
        del tmp
    return rv

cdef _get_c_func_copy_from_ve_to_gpu(data):
    # TODO: implement direct transfer and switch
    return _copy_from_ve_to_gpu_via_host

def _get_func_copy_from_ve_to_gpu(data):
    copyfunc_c = _get_c_func_copy_from_ve_to_gpu(data)
    def copy_from_ve_to_gpu(dst, src, size):
        # When data is on VE, Flush all preceding requests in order to
        # update data to be transferred.
        nlcpy.request.flush()
        return copyfunc_c(dst, src, size)
    return copy_from_ve_to_gpu

def _is_on_VH(data):
    location = data.memloc
    return location == 'memory exists on VH' or location == 'memory exists on VE and VH'

def convert_from_nlcpy_to_cupy(data):
    assert(isinstance(data, nlcpy.ndarray))
    if _is_on_VH(data):
        # data is on VH
        # simply create a cupy.ndarray from numpy.ndarray in data
        return  cupy.asarray(data.get())

    if not data._c_contiguous and data._f_contiguous:
        order = 'F'
    else:
        order = 'C'

    copyfunc = _get_func_copy_from_ve_to_gpu(data)

    # assume nlcpy.dtype == cupy.dtype
    rv = cupy.empty(data.shape, data.dtype, order)
    cdef long gpumem = rv.data.ptr
    # transfer data to rv
    if data._c_contiguous or data._f_contiguous:
        copyfunc(gpumem, data.ve_adr, data.nbytes)
    else:
        # Unless data is contiguous, create a temporary C-contiguous ndarray.
        # NLCPy does not provide ascontiguousarray(), instead,
        # ndarray.copy with default order creates C-congiguous ndarray.
        tmp = data.copy()
        copyfunc(gpumem, tmp.ve_adr, data.nbytes)
        del tmp
    return rv
