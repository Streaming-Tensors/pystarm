# import os
# import argparse
# import pyttb as ttb
# import numpy as np
# from alg import tsvdm_3way

# if __name__ == "__main__":  
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--m", type=int, help="Dimension on 1st mode")
    # parser.add_argument("-n", "--n", type=int, help="Dimension on 2nd mode")
    # parser.add_argument("-p", "--p", type=int, help="Dimension on 3rd mode")
    # args = parser.parse_args()
    # print(args)

    # m = args.m
    # n = args.n
    # p = args.p
    # A = ttb.tenrand((m,n,p))
    # # print(A)

    # dftmtx = np.fft.fft(np.eye(p))
    # # print(dftmtx)
    
    # U, S, VT = tsvdm_3way(A, dftmtx)
    # print(U.shape)
    # print(S.shape)
    # print(VT.shape)
    

    # # print(dftmtx.__array_interface__)

import numpy as np
import pystarm
from pystarm import Matrix
import time

# arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
# print(arr1)
# arr2 = np.arange(12, dtype=np.float64).reshape((3,4), order='F')
# print(arr2)
# print(np.matmul(arr1,arr2))

# mat1 = Matrix(arr1, 4, 3)
# mat2 = Matrix(arr2, 3, 4)
# mat3 = pystarm.matmul(mat1, mat2)
# pystarm.print(mat3)
# https://stackoverflow.com/questions/38410631/given-a-byte-buffer-dtype-shape-and-strides-how-to-create-numpy-ndarray
# arr3 = np.frombuffer(mat3, dtype=np.float64)
# print(arr3)
# print(arr3.shape)
# print(arr3.strides)
# # arr3[3] = 0
# arr3 = arr3.reshape((4,4), order='F')
# print(arr3)
# arr3[2,3] = 0
# pystarm.print(mat3)

arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
mat1 = pystarm.Matrix(arr1, 4, 3)
# print(mat1.getdims())
arr2 = np.frombuffer(mat1, dtype=np.float64).reshape(mat1.getdims(), order='F', copy = False)
# print(arr2)
flag = np.allclose(arr1, arr2)
