.. _basic_usage:

Basic Usage
===========

.. contents:: :local:


Preparation
-----------


If you use Vector Engine (VE) on OrchesPy, setup the environment for NLCPy.
`NLCPy: NumPy-like API accelerated with SX-Aurora TSUBASA <https://www.hpc.nec/documents/nlcpy/en/usage/basic.html>`_

If you use CUDA GPU on OrchesPy, setup the environment for CuPy.
`CuPy: NumPy & SciPy for GPU <https://docs.cupy.dev/en/stable/install.html>`_



Supported Python Versions
-------------------------

OrchesPy is available from Python version 3.8.

.. note::

    - OrchesPy is not supported for Python version 3.7 or earlier.

    - This manual is intended for users who will use Python version 3.8.


Import Package
--------------

When you use OrchesPy in your Python scripts, the package ``orchespy`` must be imported.

* When running a script using OrchesPy in interactive mode: 

    ::

        $ python
        >>> import orchespy

* When running a script using OrchesPy in non-interactive mode:

    ::

        import orchespy


Also import ``devicetype``  of ``orchespy`` to use device types.

* When running a script using OrchesPy in interactive mode: 

    ::

        $ python
        >>> import orchespy.devicetype

    or

    ::

        $ python
        >>> from orchespy.devicetype import VE

* When running a script using OrchesPy in non-interactive mode:

    ::

        import orchespy.devicetype

    or

    ::

        from orchespy.devicetype import VE


After you import ``orchespy`` successfully in a Python script, 
the script can use decorator and function of OrchesPy
described in :ref:`Reference <orchespy_reference>` .

.. note::
    If you use Vector Engine (VE) on OrchesPy,
    you need to execute the environment setup script ``nlcvars.sh``
    or ``nlcvars.csh`` once in advance.

    * When using ``sh`` or its variant:

    ::

        $ source /opt/nec/ve/nlc/X.X.X/bin/nlcvars.sh

    * When using ``csh`` or its variant:

    ::

        % source /opt/nec/ve/nlc/X.X.X/bin/nlcvars.csh

    Here, **X.X.X** denotes the version number of NEC Numeric Library Collection on your x86 node(Vector Host).


An easy example of OrchesPy script is shown below:

.. doctest::

    >>> import orchespy
    >>> from orchespy.devicetype import VE
    >>> import numpy as np
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = orchespy.transfer_array(x1, VE)
    >>> type(x2)
    <class 'nlcpy.core.core.ndarray'>

Details of usage of decorator is described in :ref:`Decoration of function <orchespy_usage_decoration>` 

Details of usage of function is described in :ref:`Function <orchespy_usage_function>` 

Device Type
--------------
OrchesPy provides "Device Type" to specify a device where you want to execute.
The current version of OrchesPy provides the following "Device Type" of the :class:`orchespy.devicetype` class:

================================ =========================================================
Class Name                       Device Type
================================ =========================================================
Host                             x86 node
VE                               Vector Engine
CUDAGPU                          CUDA GPU
================================ =========================================================


