Overview
==========

OrchesPy is a Python library for numerical programs working
on the top of NumPy or NumPy-like libraries for accelerators
in a heterogeneous system.
OrchesPy facilitates porting a NumPy program to a heterogeneous system
and reduces overheads of data transfer between accelerators.

Heterogeneous computing, using different types of computational units
in a system, has been popular for performance and power efficiency.
There are various accelerators for computing: GPU (NVIDIA, AMD, etc),
Vector Engine (NEC), FPGA, etc.
Accelerators have different characteristics in performance.
For better performance, a system can have multiple types of
accelerators with different performance characteristics.

It requires many modifications to port a program on such a
heterogeneous system with multiple types of accelerators.
A programmer needs to use different libraries for different
accelerators.
For example, to port a NumPy program to a system with NVIDIA CUDA GPU
and NEC SX-Aurora TSUBASA, a program uses NumPy on host part, CuPy on
GPU part and NLCPy on VE part.
A programmer also needs to add data transfer code between parts
running on different devices.

We developed OrchesPy, another NumPy-like library for heterogeneous
computing to make it easier to port and optimize a NumPy program on
a heterogeneous system with multiple types of accelerators.
The name "OrchesPy" comes from that the library **orches**\ trates
NumPy-like **Py**\ thon libraries for various devices.

OrchesPy provides following two features:

* "device" directive-like decorator converting N-dimension arrays
  (array_like) in arguments transparently in order to execute code on
  a specified device, and
* "transfer_array" function to transfer an array_like from a device
  to another device explicitly in portable way.

