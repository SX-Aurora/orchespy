from .devicetype.find_device_class import find_device_class


def _get_contiguous(obj):
    if hasattr(obj, 'flags'):
        if obj.flags.c_contiguous:
            return "C"
        elif obj.flags.f_contiguous:
            return "F"
        else:
            return None
    else:
        if obj._c_contiguous:
            return "C"
        elif obj._f_contiguous:
            return "F"
        else:
            return None


def _check_contiguous(dst, src):
    dst_order = _get_contiguous(dst)
    if dst_order is None:
        raise ValueError('dst needs to be C or F contiguous.')

    src_order = _get_contiguous(src)
    if src_order is None:
        return
    if dst_order != src_order:
        raise ValueError('dst and src contiguous must match.')
    return


def transfer_array(src, target):
    """Transfer N-dimension array to a specified target device.

    Parameters
    ----------
    src : array_like
        N-dimension array on a device or host to be transferred.
    target : devicetype
        Target device; device to be transferred x to.
        Specify the devicetype class that corresponds to the device.
        See :class:`orchespy.devicetype` for what you can specify.

    Returns
    -------
    ndarray:
        N-dimension array of `src` on the `target` device.

    See Also
    --------
    orchespy.devicetype: Supported device types.

    Examples
    --------
      Transfer ndarray on a host to CUDAGPU.

      >>> import orchespy
      >>> import orchespy.devicetype
      >>> import numpy
      >>> import cupy
      >>> x = numpy.asarray([1, 2, 3])
      >>> y = orchespy.transfer_array(x, orchespy.devicetype.CUDAGPU())
      >>> y
      array([1, 2, 3])
      >>> isinstance(y, cupy.ndarray)
      True

      If there are multiple identical devices,
      they can be specified by writing as follows.
      Also, when processing the ndarray output by transfer_array,
      it is necessary to switch to the output device after transfer.

      >>> from orchespy.devicetype import VE
      >>> import orchespy
      >>> import numpy as np
      >>> import nlcpy as vp
      >>>
      >>> src = np.ones((2, 2))
      >>> x = orchespy.transfer_array(src, VE(1))
      >>> x.venode.id
      1
      >>> vp.venode.VE(1).use()
      <VE node logical_id=1, physical_id=1>
      >>> y = vp.ones((2, 2))
      >>> mul = x * y
      >>> mul.venode.id
      1
      >>> type(mul)
      <class 'nlcpy.core.core.ndarray'>
    """
    if isinstance(target, type):
        target = target()
    if not hasattr(target, 'numpy_class'):
        raise ValueError('Assign a device class to target.')

    if target.can_transfer(src):
        dst = target.create_ndarray_on_device(src)
        target.transfer_array_content(dst, src)
        return dst
    else:
        srctype = find_device_class(src)
        if srctype is not None:
            srcdev = srctype.get_device(src)
            if srcdev.can_transfer_to(src, target):
                dst = target.create_ndarray_on_device(src)
                srcdev.transfer_array_content_to(dst, src)
                return dst
            else:
                raise ValueError('This src or target cannot be transferred.')
    return src


def transfer_array_content(dst, src):
    """Transfer N-dimension array to a specified N-dimension array.

    Parameters
    ----------
    dst : array_like
        N-dimension array on a device or host that receives the src value.
    src : array_like
        N-dimension array on a device or host to be transferred.

    Returns
    -------
    None

    Examples
    --------
      Transfer ndarray on a host to CUDAGPU.

      >>> import orchespy
      >>> import numpy
      >>> import cupy
      >>> x = numpy.asarray([1, 2, 3])
      >>> y = cupy.asarray([-1, -1, -1])
      >>> orchespy.transfer_array_content(y, x)
      >>> y
      array([1, 2, 3])
      >>> isinstance(y, cupy.ndarray)
      True

      Below is an example of an environment with multiple identical devices.

      >>> import orchespy
      >>> import numpy as np
      >>> import nlcpy as vp
      >>>
      >>> src = np.ones((2, 2), dtype='i8', order='C') * 3
      >>> vp.venode.VE(1).use()
      <VE node logical_id=1, physical_id=1>
      >>> dst = vp.zeros((2, 2), dtype='i8', order='C')
      >>> orchespy.transfer_array_content(dst, src)
      >>> dst.venode.id
      1
      >>> type(dst)
      <class 'nlcpy.core.core.ndarray'>
      >>> dst
      array([[3, 3],
             [3, 3]])
    """
    dsttype = find_device_class(dst)
    if dsttype is None:
        raise ValueError('The device could not be found'
                         ' from the first argument.')

    dstdev = dsttype.get_device(dst)
    if dstdev is not None:
        if dstdev.can_transfer(src):
            if dst.shape != src.shape:
                raise ValueError('Shape mismatch.')
            if dst.dtype != src.dtype:
                raise ValueError('dtype mismatch.')
            _check_contiguous(dst, src)
            dstdev.transfer_array_content(dst, src)
            return
        else:
            srctype = find_device_class(src)
            if srctype is None:
                raise ValueError('The device could not be found'
                                 ' from the second argument.')

            srcdev = srctype.get_device(src)
            if srcdev.can_transfer_to(src, dstdev):
                if dst.shape != src.shape:
                    raise ValueError('Shape mismatch.')
                if dst.dtype != src.dtype:
                    raise ValueError('dtype mismatch.')
                _check_contiguous(dst, src)
                srcdev.transfer_array_content_to(dst, src)
                return
            else:
                raise ValueError('Objects that cannot be transferred.')
