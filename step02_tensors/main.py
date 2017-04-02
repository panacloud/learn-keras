import numpy as np

#Scalars (0D tensors)
a = np.array(10)
print("Scalars (0D tensors)")
print(a)
print(a.dtype)
print(a.ndim)
print(a.shape)

#Vectors (1D tensors)
b = np.array([1, 2, 3])
print("Vectors (1D tensors)")
print(b)
print(b.dtype)
print(b.ndim)
print(b.shape)

#Matrices (2D tensors)
c = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])
print("Matrices (2D tensors)")
print(c)
print(c.dtype)
print(c.ndim)
print(c.shape)

#3D tensors
d = np.array([[[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            [[10, 11, 12],
             [13, 14, 15],
             [16, 17, 18]]])
print("3D tensors")
print(d)
print(d.dtype)
print(d.ndim)
print(d.shape)




