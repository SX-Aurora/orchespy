_ndarray_to_devicetype = {}


def register_devicetype(devtype, arraytype):
    _ndarray_to_devicetype[arraytype] = devtype


def _find_devicetype(array):
    return _ndarray_to_devicetype.get(type(array))


def transfer_array(x, target):
    """Transfer N-dimension array to a specified target device.

    Parameters
    ----------
    x : array_like
        N-dimension array on a device or host to be transferred.
    target : devicetype
        Target device; device to be transferred x to.
        Specify the devicetype class that corresponds to the device.
        See :class:`orchespy.devicetype` for what you can specify.

    Returns
    -------
    ndarray:
        N-dimension array of `x` on the `target` device.

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
    """
    if isinstance(target, type):
        target = target()

    if isinstance(x, target.numpy_class.ndarray):
        # TODO: check whether x is on target or not.
        return x
    # find the device type of source
    srctype = _find_devicetype(x)
    if srctype is None:
        return x
    source = srctype.find_device(x)
    if source is None:
        return x

    # get a transfer function from the source and target device types
    xferfunc = target.func_to_transfer_ndarray_from(source)
    if callable(xferfunc):
        return xferfunc(x)

    xferfunc = source.func_to_transfer_ndarray_to(target)
    if callable(xferfunc):
        return xferfunc(x)

    # transfer via Host
    array_on_host = source.get_ndarray_on_host(x)
    return target.get_ndarray_on_device(array_on_host)
