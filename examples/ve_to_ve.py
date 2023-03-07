import nlcpy, orchespy
from orchespy.devicetype import VE

x1 = nlcpy.arange(9.0).reshape((3, 3))
x2 = orchespy.transfer_array(x1, VE(0))

print(type(x2), x2)

x1 = nlcpy.arange(9.0).reshape((3, 3))
x2 = orchespy.transfer_array(x1, VE(0))

print(type(x2), x2)
