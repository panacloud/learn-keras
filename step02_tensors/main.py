import numpy as np

#Scalars (0D tensors)
a = np.array(10)
print(a)
print(a.ndim)

#Vectors (1D tensors)
b = np.array([1, 2, 3])
print(b)
print(b.ndim)

#Matrices (2D tensors)
c = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])
print(c)
print(c.ndim)

#3D tensors
d = np.array([[[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            [[10, 11, 12],
             [13, 14, 15],
             [16, 17, 18]]])
print(d)
print(d.ndim)




