from orchespy import device, transfer_array
from orchespy.devicetype import Host, CUDAGPU

import numpy
import cupy


@device(Host)
def sum_host(x, y):
    return x + y


@device(CUDAGPU)
def sum_dev(x, y):
    return x + y


@device(Host, numpy_module_arg='xp')
def create_host(size, xp):
    return xp.random.rand(*size)


@device(CUDAGPU, numpy_module_arg='xp')
def create_dev(size, xp):
    return xp.random.rand(*size)


size = (1000, 500, 500)

x1 = create_host(size)
x2 = create_dev(size)
x3 = create_dev(size)
print(type(x1), type(x2), type(x3))

y1 = sum_dev(x1, x2)
z1d = sum_dev(y1, x3)

y2 = sum_host(x1, x2)
z2h = sum_host(y2, x3)

z1h = transfer_array(z1d, Host())
diffh = z2h - z1h

print('Norm on host:', numpy.linalg.norm(diffh))

z2d = transfer_array(z2h, CUDAGPU)

diffd = z2d - z1d
print('Norm on device:', cupy.linalg.norm(diffd))
