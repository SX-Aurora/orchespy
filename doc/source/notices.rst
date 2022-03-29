Notices and Restrictions
========================

.. contents:: :local:

This page describes notices and restrictions of OrchesPy.

Notices
-------

There are some functions which some NumPy-like packages implement
but are missing in other NumPy-like packages.
A program using such functions

Arguments passed as only array_like objects are converted on
entering a decorated function; however, arguments passed as
objects including array_like in attributes are not converted.


Limitations
-----------

The current OrchesPy supports one device for each device type.
Even if multiple VEs and GPUs are connected, an OrchesPy program
can use one VE and one GPU.
In future version, multiple devices will be supported.
