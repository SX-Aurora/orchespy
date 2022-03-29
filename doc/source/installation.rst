.. _installation:

Installation
============

This page describes installation of OrchesPy.


Requirements
------------

Before the installation of OrchesPy, the following components are required to be
installed.

* | `Python <https://www.python.org/>`_

    - required version: 3.8

* | `NumPy <https://www.numpy.org/>`_

    - required version: v1.17, v1.18, v1.19, or v1.20

* | `NLCPy <https://www.hpc.nec/documents/nlcpy/en/>`_

    - To run programs on VE, NLCPy and its dependencies such as veoffload are required.
    - required version: v2.0.1

* | `CuPy <https://cupy.dev/>`_

    - To run programs on CUDA GPU, CuPy and its dependencies such as CUDA toolkit working with your GPU
      are required



Install from wheel
------------------

You can install OrchesPy by executing either of following commands.

* Install from PyPI

  ::

      $ pip install orchespy


* Install from your local computer

    1. Download the wheel package from `GitHub <https://github.com/SX-Aurora/orchespy/>`_.
    2. Put the wheel package to your any directory.
    3. Install the local wheel package via pip command.

       ::

           $ pip install <path_to_wheel>


Install from source (with building)
-----------------------------------

To build OrchesPy, install CUDA toolkit and veoffload (VEO) for CuPy and NLCPy.
Download the source tree from GitHub.

::

    $ git clone https://github.com/SX-Aurora/orchespy.git
    $ cd orchespy
    $ pip install .


