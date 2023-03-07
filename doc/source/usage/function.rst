.. _orchespy_usage_function:

Function
=========

.. contents:: :local:
   :depth: 1


Transfer array
------------------

OrchesPy provides a function to transfer an ndarray to a specified device in a portable way.

* Example 1:

.. doctest::

      >>> import orchespy
      >>> import orchespy.devicetype
      >>> import nlcpy
      >>> import cupy
      >>> x = nlcpy.asarray([1, 2, 3])
      >>> y = orchespy.transfer_array(x, orchespy.devicetype.CUDAGPU())
      >>> y
      array([1, 2, 3])
      >>> isinstance(y, cupy.ndarray)
      True

An NLCPy ndarray ``x`` is transferred to GPU as a CuPy ndarray


* Example 2:

.. doctest::

      >>> import orchespy
      >>> import orchespy.devicetype
      >>> import numpy
      >>> 
      >>> dev = orchespy.devicetype.VE
      >>> @orchespy.device(dev, numpy_module_arg='xp')
      ... def create_on_dev(xp):
      ...    return xp.arange(1, 6)
      >>> 
      >>> x = create_on_dev()
      >>> print(type(x))
      <class 'nlcpy.core.core.ndarray'>
      >>> 
      >>> y = numpy.arange(6, 11)
      >>> y_dev = orchespy.transfer_array(y, dev)
      >>> 
      >>> print(x + y_dev)
      [ 7  9 11 13 15]


An NumPy ndarray ``y`` is transferred to VE as a NLCPy ndarray.
You can switch the data transfer to GPU 
just by changing ``dev`` to "orchespy.devicetype.CUDAGPU".


* Example 3:

.. doctest::

      >>> import orchespy
      >>> from orchespy.devicetype import VE
      >>> import numpy as np
      >>> x1 = np.arange(9.0).reshape((3, 3))
      >>> x2 = orchespy.transfer_array(x1, VE(1))
      >>> type(x2)
      <class 'nlcpy.core.core.ndarray'>
      >>> x2.venode.id
      1

If you have multiple identical devices, you can specify any device.

Transfer array content
------------------------


You can transfer an array-like to a specified array-like.

* Example:

.. doctest::

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


An NumPy ndarray ``x`` is transferred to GPU as a CuPy ndarray.
When the transfer destination is VE, it is transferred as NLCPy ndarray.
It also enables transfer from VE to GPU and transfer from GPU to VE.
