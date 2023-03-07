from orchespy import device, transfer_array
from orchespy.devicetype import Host, VE
import orchespy.numpy as np

import numpy
import nlcpy


@device(Host)
def sum_host(x, y):
    return x + y


@device(VE)
def sum_dev(x, y):
    return x + y


@device(Host)
def create_host(size):
    return np.random.rand(*size)


@device(VE)
def create_dev(size):
    return np.random.rand(*size)


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

z2d = transfer_array(z2h, VE)

diffd = z2d - z1d
print('Norm on device:', nlcpy.linalg.norm(diffd))

@device(VE(0))
def sum_dev0(x, y):
    return x + y

@device(VE(0))
def create_dev0(size):
    return np.random.rand(*size)

x2 = create_dev0(size)
x3 = create_dev0(size)

y1 = sum_dev(x1, x2)
z1d = sum_dev(y1, x3)

y2 = sum_host(x1, x2)
z2h = sum_host(y2, x3)

z2d = transfer_array(z2h, VE(0))

diffd = z2d - z1d
print('Norm on device(0):', nlcpy.linalg.norm(diffd))
