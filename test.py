import numpy as np

A = np.array([[1,2,3],
               [1,0,3],
               [1,2,3]])

B = np.array([[0,2,3],
              [0,2,3],
              [0,2,3]])

A = np.where(A<2, 6, A)
B = np.where(A<2, 7, B)

print(B)