.. _orchespy_usage_decoration:

Decoration of function
========================================


OrchesPy provides a decorator :func:`orchespy.decorator`. 
``orchespy.device()`` enables to decorate a function to execute
on a device you want.

* Example 1:

.. doctest::

    >>> from orchespy import device
    >>> from orchespy.devicetype import CUDAGPU
    >>> import numpy as np
    >>>
    >>> @device(CUDAGPU)
    ... def exec_on_gpu(x, y):
    ...     return x * y
    >>>
    >>> x = np.linspace(1, 5, 5)
    >>> y = np.linspace(1, 5, 5)
    >>> z = exec_on_gpu(x, y)
    >>> type(z)   # doctest: +SKIP
    <class 'cupy.core.core.ndarray'>   # ndarray of nlcpy

``x`` and ``y`` are ``numpy.ndarray``. The function exec_on_gpu() will execute on CUDA GPU.
You don't need to convert x and y to ``cupy.ndarray``.
OrchesPy will automatically convert them to the target ndarray type.


.. note::
    Since the function exec_on_gpu() is executed on CUDA GPU, an ndarray of CuPy is returned. 
    The data of ndarray is transferred automatically to CUDA GPU.


* Example 2:

.. doctest::

    >>> from orchespy import device
    >>> from orchespy.devicetype import VE
    >>>
    >>> @device(VE, numpy_module_arg='xp')
    ... def create_array_on_VE(xp):
    ...     return xp.random.rand(2, 2)
    >>>
    >>> x = create_array_on_VE()
    >>> type(x)   # doctest: +SKIP
    <class 'nlcpy.core.core.ndarray'>   # ndarray of nlcpy

You can pass the package of NumPy-like library for a device to a decorated function.
In Example 2, the argument ``xp`` is set to ``nlcpy``.
``nlcpy.random.rand()``, the array creation function in NLCPy, is called
at the line ``xp.random.rand()``.

