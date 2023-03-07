# distutils: language = c++
# Translation module between CuPy on NVIDIA GPU and NLCPy on NEC Vector Engine
import nlcpy
import nlcpy.core
import nlcpy.core.core

from nlcpy.venode import VE

import cupy

from libc.stdlib cimport *  # NOQA
from libc.stdint cimport *

import os
from collections import OrderedDict  # for LRU Cache

cdef extern from "<errno.h>":
    int EBUSY, errno

cdef extern from '<cuda_runtime.h>' nogil:
    cdef enum cudaError:
        cudaSuccess
    ctypedef cudaError cudaError_t
    cdef enum cudaMemcpyKind:
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice

    cdef cudaError_t cudaMemcpy(void *, const void *, size_t,
                                cudaMemcpyKind) nogil

cdef extern from "<ve_offload.h>" nogil:
    cdef int veo_hmemcpy(void *, const void *, size_t)

# use native transfer library
cdef _copy_from_ve_to_gpu_via_host(dst, src):
    # src : nlcpy
    cdef size_t size = src.nbytes
    cdef uint64_t src_addr = src.veo_hmem
    # dst : cupy
    cdef uint64_t dst_addr = dst.data.ptr

    cdef void *buf = malloc(size)
    cdef int rv
    try:
        # VE -> Host
        rv = veo_hmemcpy(buf, <void *>src_addr, size)
        if rv != 0:
            raise RuntimeError("veo_hmemcpy failed")
        # Host -> GPU
        rv = cudaMemcpy(<void *>dst_addr, <void *>buf, size,
                        cudaMemcpyHostToDevice)
        if rv != cudaSuccess:
            raise RuntimeError("cudaMemcpy failed")
    finally:
        free(buf)

# use native transfer library
cdef _copy_from_gpu_to_ve_via_host(dst, src):
    # src : cupy
    cdef size_t size = src.nbytes
    cdef uint64_t src_addr = src.data.ptr
    # dst : nlcpy
    cdef uint64_t dst_addr = dst.veo_hmem

    cdef void *buf = malloc(size)
    cdef int rv
    try:
        # GPU -> Host
        rv = cudaMemcpy(<void *>buf, <void *>src_addr, size,
                        cudaMemcpyDeviceToHost)
        if rv != cudaSuccess:
            raise RuntimeError("cudaMemcpy failed")
        # Host -> VE
        rv = veo_hmemcpy(<void *>dst_addr, buf, size)
        if rv != 0:
            raise RuntimeError("veo_hmemcpy failed")
    finally:
        free(buf)

# ---------------------------------------------------------------------------------------
cdef extern from '<interdevcopy.h>' nogil:
    cdef enum interdevcopy_memory_type:
        INTERDEVCOPY_MEMORY_HOST_MEM
        INTERDEVCOPY_MEMORY_VE_HMEM
        INTERDEVCOPY_MEMORY_CUDA_MEM
    cdef cppclass interdevcopy_memory_region
    cdef cppclass interdevcopy_channel

cdef extern from "<dlfcn.h>" nogil:
    void *dlopen(const char *, int)
    char *dlerror()
    void *dlsym(void *, const char *)
    int dlclose(void *)
    int RTLD_LAZY
    int RTLD_NOW
    int RTLD_GLOBAL
    int RTLD_LOCAL

cdef void* (*hook_interdevcopy_create_memory_region)(void *,
                                                     size_t,
                                                     interdevcopy_memory_type,
                                                     void *)
cdef int(*hook_interdevcopy_destroy_memory_region)(void *)
cdef void* (*hook_interdevcopy_create_channel)(void *, void *, void *)
cdef int(*hook_interdevcopy_destroy_channel)(void *)
cdef ssize_t(*hook_interdevcopy_copy)(void *, unsigned long, unsigned long,
                                      size_t, void *)


def get_interdevcopy_path():
    return '/opt/nec/interdevcopy/lib64/libinterdevcopy.so.1'


cdef int _get_interdevcopy_sym() except -1:
    global hook_interdevcopy_create_memory_region
    global hook_interdevcopy_destroy_memory_region
    global hook_interdevcopy_create_channel
    global hook_interdevcopy_destroy_channel
    global hook_interdevcopy_copy

    cdef void *hdl_interdevcopy = NULL
    cdef char *err = NULL
    lib_file = bytes(get_interdevcopy_path(), 'utf-8')

    # When succesfully loaded.
    if hook_interdevcopy_copy != NULL:
        return 0

    if not os.path.isfile(lib_file):
        return 1

    hdl_interdevcopy = <void *>dlopen(lib_file, RTLD_NOW)
    err = dlerror()
    if err != NULL:
        raise IOError("dlopen failed =%s" % err)

    hook_interdevcopy_create_memory_region = \
        <void* (*)(void *, size_t, interdevcopy_memory_type, void *)>\
        dlsym(hdl_interdevcopy, 'interdevcopy_create_memory_region')
    err = dlerror()
    if err != NULL:
        dlclose(hdl_interdevcopy)
        raise IOError("interdevcopy_create_memory_region failed =%s" % err)

    hook_interdevcopy_destroy_memory_region = <int(*)(void *)>\
        dlsym(hdl_interdevcopy, 'interdevcopy_destroy_memory_region')
    err = dlerror()
    if err != NULL:
        dlclose(hdl_interdevcopy)
        raise IOError("interdevcopy_destroy_memory_region failed =%s" % err)

    hook_interdevcopy_create_channel = <void * (*)(void *, void *, void *)>\
        dlsym(hdl_interdevcopy, 'interdevcopy_create_channel')
    err = dlerror()
    if err != NULL:
        dlclose(hdl_interdevcopy)
        raise IOError("interdevcopy_create_channel failed =%s" % err)

    hook_interdevcopy_destroy_channel =\
        <int(*)(void *)>dlsym(hdl_interdevcopy, 'interdevcopy_destroy_channel')
    err = dlerror()
    if err != NULL:
        dlclose(hdl_interdevcopy)
        raise IOError("interdevcopy_destroy_channel failed =%s" % err)

    hook_interdevcopy_copy =\
        <ssize_t(*)(void *, unsigned long, unsigned long, size_t, void *)>\
        dlsym(hdl_interdevcopy, 'interdevcopy_copy')
    err = dlerror()
    if err != NULL:
        dlclose(hdl_interdevcopy)
        raise IOError("interdevcopy_copy failed =%s" % err)

    return 0

# interdevcopy_memory_region_cache_max should be set to
# more than twice that of interdevcopy_channel_cache_max
cdef int interdevcopy_memory_region_cache_max = 32 + 2  # Reserve
cdef int interdevcopy_channel_cache_max = 16
interdevcopy_memory_region_cache = OrderedDict()
interdevcopy_channel_cache = OrderedDict()
cdef interdevcopy_memory_region* _get_interdevcopy_cache_memory_region(array
                                                                       ) except NULL:
    cdef int ret = 0
    cdef interdevcopy_memory_region* memory = NULL
    cdef uint64_t addr = 0
    cdef size_t size = 0
    cdef uint64_t chached_interdevcopy_memory_region = 0
    cdef uint64_t tmp_interdevcopy_memory_region = 0
    cdef uint64_t tmp_interdevcopy_channel = 0

    # interdevcopy_memory_region from LRU Cache objects exist check
    if isinstance(array, nlcpy.ndarray):
        addr = array.veo_hmem
    else:
        addr = array.data.ptr
    size = array.nbytes
    if len(interdevcopy_memory_region_cache) > 0:
        val = interdevcopy_memory_region_cache.get((addr, size))
        if val is not None:
            chached_interdevcopy_memory_region = val
            interdevcopy_memory_region_cache.move_to_end((addr, size), last=True)
            return <interdevcopy_memory_region *>chached_interdevcopy_memory_region

    # Cache New interdevcopy_memory_region
    if isinstance(array, nlcpy.ndarray):
        size = array.nbytes
        memory = <interdevcopy_memory_region *>hook_interdevcopy_create_memory_region(
            <void *>addr, size, INTERDEVCOPY_MEMORY_VE_HMEM, <void *>0)
    elif isinstance(array, cupy.ndarray):
        size = array.nbytes
        memory = <interdevcopy_memory_region *>hook_interdevcopy_create_memory_region(
            <void *>addr, size, INTERDEVCOPY_MEMORY_CUDA_MEM, <void *>0)
    else:
        raise IOError("get_interdevcopy_cache_memory_region not nlcpy/cupy faild.")

    if <int64_t>memory < 0:
        raise IOError("interdevcopy_create_memory_region faild. err= %d"
                      % <int64_t>memory)

    if len(interdevcopy_memory_region_cache) >= interdevcopy_memory_region_cache_max:
        _ml = list(interdevcopy_memory_region_cache.items())
        _pop_key = None
        for cache in _ml:
            tmp_interdevcopy_memory_region = cache[1]
            ret = hook_interdevcopy_destroy_memory_region(
                <interdevcopy_memory_region *>tmp_interdevcopy_memory_region)
            if ret == -EBUSY:
                continue
            elif ret == 0:
                _pop_key = cache[0]
                break
            elif ret < 0:
                raise IOError("interdevcopy_destroy_memory_region faild. err=%d" % ret)
        # Never removed
        if _pop_key is None:
            raise IOError("interdevcopy_destroy_memory_region faild. err=%d" % ret)
        interdevcopy_memory_region_cache.pop(_pop_key)
    interdevcopy_memory_region_cache[(addr, array.nbytes)] = <uint64_t>memory

    return memory


cdef interdevcopy_channel* _get_interdevcopy_cache_channel(
        interdevcopy_memory_region *dst, interdevcopy_memory_region *src) except NULL:
    cdef int ret = 0
    cdef interdevcopy_channel* channel = NULL
    cdef uint64_t tmp_interdevcopy_channel = 0
    # interdevcopy_channel from LRU Cache objects
    if len(interdevcopy_channel_cache) > 0:
        key = (<uint64_t>dst, <uint64_t>src)
        tmp = interdevcopy_channel_cache.get(key)
        if tmp is not None:
            interdevcopy_channel_cache.move_to_end(key, last=True)
            tmp_interdevcopy_channel = tmp
            return <interdevcopy_channel *>tmp_interdevcopy_channel

    # Cache New interdevcopy_channel
    channel = <interdevcopy_channel *>hook_interdevcopy_create_channel(
        dst, src, <void *>0)
    if <int64_t>channel < 0:
        _cl = list(interdevcopy_channel_cache.items())
        for chache in _cl:
            tmp = interdevcopy_channel_cache.popitem(False)
            tmp_interdevcopy_channel = tmp[1]
            ret = hook_interdevcopy_destroy_channel(
                <interdevcopy_channel *>tmp_interdevcopy_channel)
            if ret < 0:
                raise IOError("interdevcopy_destroy_channel faild. err=%d" % ret)
            channel = <interdevcopy_channel *>hook_interdevcopy_create_channel(
                dst, src, <void *>0)
            if <int64_t>channel < 0:
                continue
            else:
                break
        else:
            raise IOError("interdevcopy_create_channel faild. err=%d" % <int64_t>channel)

    if len(interdevcopy_channel_cache) >= interdevcopy_channel_cache_max:
        tmp = interdevcopy_channel_cache.popitem(False)
        tmp_interdevcopy_channel = tmp[1]
        ret = hook_interdevcopy_destroy_channel(
            <interdevcopy_channel *>tmp_interdevcopy_channel)
        if ret < 0:
            raise IOError("interdevcopy_destroy_channel faild. err=%d" % ret)
    interdevcopy_channel_cache[(<uint64_t>dst, <uint64_t>src)] = <uint64_t>channel

    return channel

cdef void _copy_interdevcopy_direct(dst, src) except *:
    cdef interdevcopy_memory_region* src_memrgn
    cdef interdevcopy_memory_region* dst_memrgn
    cdef interdevcopy_channel* channel
    cdef ssize_t transfer_size = 0
    cdef size_t size = src.nbytes

    src_memrgn = _get_interdevcopy_cache_memory_region(src)
    dst_memrgn = _get_interdevcopy_cache_memory_region(dst)
    channel = _get_interdevcopy_cache_channel(dst_memrgn, src_memrgn)

    transfer_size = hook_interdevcopy_copy(channel, 0, 0, size, <void *>0)
    if transfer_size < 0:
        raise IOError("interdevcopy_copy faild. err=%d" % transfer_size)

cdef bint _is_interdevcopy_library_available():
    # check interdevcopy library.
    if _get_interdevcopy_sym() == 0:
        return True
    return False

# ---------------------------------------------------------------------------------------
cdef unsigned long data_size_threshold = 0


def set_data_size_threshold(threshold):
    global data_size_threshold
    data_size_threshold = int(threshold)


cdef bint _is_enable_direct_transfer(data):
    # Data size threshold.
    # src.nbytes >= data_size_threshold is true.
    if data.nbytes < data_size_threshold:
        return False

    # interdevcopy library available
    if _is_interdevcopy_library_available():
        return True

    return False


cdef _get_func_copy_from_gpu_to_ve(data):
    if _is_enable_direct_transfer(data):
        return _copy_interdevcopy_direct
    else:
        return _copy_from_gpu_to_ve_via_host


def convert_from_cupy_to_nlcpy(dst, data):
    if not data._c_contiguous and data._f_contiguous:
        order = 'F'
    else:
        if not dst._c_contiguous and dst._f_contiguous:
            order = 'F'
        else:
            order = 'C'

    copyfunc = _get_func_copy_from_gpu_to_ve(data)

    # Flush all requests for ndarray rv to be allocated
    dst.venode.synchronize()

    if data._c_contiguous or data._f_contiguous:
        copyfunc(dst, data)
    else:
        # Unless data is contiguous, create a temporary C-contiguous ndarray.
        tmp = data.copy(order=order)
        copyfunc(dst, tmp)
        del tmp


cdef _get_c_func_copy_from_ve_to_gpu(data):
    if _is_enable_direct_transfer(data):
        return _copy_interdevcopy_direct
    else:
        return _copy_from_ve_to_gpu_via_host


def _get_func_copy_from_ve_to_gpu(data):
    copyfunc_c = _get_c_func_copy_from_ve_to_gpu(data)

    def copy_from_ve_to_gpu(dst, src):
        # When data is on VE, Flush all preceding requests in order to
        # update data to be transferred.
        src.venode.synchronize()
        return copyfunc_c(dst, src)
    return copy_from_ve_to_gpu


def convert_from_nlcpy_to_cupy(dst, data):
    if not data._c_contiguous and data._f_contiguous:
        order = 'F'
    else:
        if not dst._c_contiguous and dst._f_contiguous:
            order = 'F'
        else:
            order = 'C'

    copyfunc = _get_func_copy_from_ve_to_gpu(data)

    if data._c_contiguous or data._f_contiguous:
        copyfunc(dst, data)
    else:
        # Unless data is contiguous, create a temporary C-contiguous ndarray.
        # NLCPy does not provide ascontiguousarray(), instead,
        # ndarray.copy with default order creates C-congiguous ndarray.
        tmp = data.copy(order=order)
        copyfunc(dst, tmp)
        del tmp
