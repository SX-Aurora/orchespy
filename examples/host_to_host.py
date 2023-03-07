import numpy, orchespy
from orchespy.devicetype import Host

x1 = numpy.arange(9.0).reshape((3, 3))
x2 = orchespy.transfer_array(x1, Host)

print(type(x2), x2)

x1 = numpy.arange(9.0).reshape((3, 3))
x2 = orchespy.transfer_array(x1, Host)

print(type(x2), x2)
