import nlcpy, orchespy
a = nlcpy.arange(9, dtype=int).reshape(3,3)
b = nlcpy.zeros((3,3), dtype=int)
orchespy.transfer_array_content(b, a)

print(type(b), b)
