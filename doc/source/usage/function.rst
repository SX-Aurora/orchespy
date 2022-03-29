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


