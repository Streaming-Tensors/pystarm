# import numpy as np
# import time

# x = np.arange(0, 1*1000*1000, 1, dtype=int)
# y = np.array(x).reshape( (1, 1000, 1000), order='C')
# # zz = np.array(x).reshape( (1000, 1000, 1000), order='F')
# z = np.array(y, order='F')
# # print(y, y.flags)
# # print(z, z.flags)

# t0 = time.time()
# Uy, Sy, Vhy = np.linalg.svd(y, full_matrices=True)
# t1 = time.time()
# Uz, Sz, Vhz = np.linalg.svd(z, full_matrices=True)
# t2 = time.time()

# print(t1-t0, t2-t1)

# # print(Uy.shape, Sy.shape, Vhy.shape)
# # print(Uy, Sy, Vhy)
# # print(Uz.shape, Sz.shape, Vhz.shape)
# # print(Uz, Sz, Vhz)


import unittest
import pystarm
import numpy as np
import pyttb
import os

class TensorTestCase(unittest.TestCase):
    def test_tensor_creation(self):
        """Test for tensor creation"""
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)
        arr2 = np.frombuffer(ten1, dtype=np.float64).reshape(ten1.getdims(), order='F', copy = False)
        flag = np.allclose(arr1, arr2)
        self.assertEqual(flag, True)

    def test_ttm(self):
        """Test ttm in the first dimension"""
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        mat_nelm = 3*5 # 15 elements
        mat_dims = (3, 5)
        mat_ndim = len(ten_dims)
        arr2 = np.arange(mat_nelm, dtype=np.float64).reshape(mat_dims, order='F')
        mat2 = pystarm.Matrix(arr2, mat_dims[0], mat_dims[1])
        
        ten2 = pystarm.ttm(ten1, mat2, 0) # On the first mode (0+1th mode)

        # ten3 = ttb.tensor(arr1)

        arr3 = np.frombuffer(ten2, dtype=np.float64).reshape(ten2.getdims(), order='F', copy = False)
        flag = np.allclose(arr1, arr3)
        self.assertEqual(flag, True)

    # def test_matmul(self):
        # """Test matrix multiplication"""
        # arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        # arr2 = np.arange(12, dtype=np.float64).reshape((3,4), order='F')
        # arr3 = np.matmul(arr1, arr2)
        # mat1 = pystarm.Matrix(arr1, 4, 3)
        # mat2 = pystarm.Matrix(arr2, 3, 4)
        # mat3 = pystarm.matmul(mat1, mat2)
        # arr4 = np.frombuffer(mat3, dtype=np.float64).reshape(mat3.getdims(), order='F', copy = False)
        # flag = np.allclose(arr3, arr4)
        # self.assertEqual(flag, True)

class MatrixTestCase(unittest.TestCase):
    def test_matrix_creation(self):
        """Test for matrix creation"""
        arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        mat1 = pystarm.Matrix(arr1, 4, 3)
        arr2 = np.frombuffer(mat1, dtype=np.float64).reshape(mat1.getdims(), order='F', copy = False)
        flag = np.allclose(arr1, arr2)
        self.assertEqual(flag, True)

    def test_matrix_update_in_python(self):
        """Test if update to the matrix on Python side gets reflected on the C++ side"""
        arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        mat1 = pystarm.Matrix(arr1, 4, 3)
        arr1[3,2] = 0 # Set 0 to the element at index(3,2)
        arr2 = np.frombuffer(mat1, dtype=np.float64).reshape(mat1.getdims(), order='F', copy = False)
        flag = np.allclose(arr1, arr2)
        self.assertEqual(flag, True)

    def test_matrix_update_in_cpp(self):
        """Test if update to the matrix in C++ side gets reflected on the Python side"""
        arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        mat1 = pystarm.Matrix(arr1, 4, 3)
        mat1.set(3,2,0) # Set 0 to the element at index (3,2)
        arr2 = np.frombuffer(mat1, dtype=np.float64).reshape(mat1.getdims(), order='F', copy = False)
        flag = np.allclose(arr1, arr2)
        self.assertEqual(flag, True)

    def test_matmul(self):
        """Test matrix multiplication"""
        arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        arr2 = np.arange(12, dtype=np.float64).reshape((3,4), order='F')
        arr3 = np.matmul(arr1, arr2, order='F')
        mat1 = pystarm.Matrix(arr1, 4, 3)
        mat2 = pystarm.Matrix(arr2, 3, 4)
        mat3 = pystarm.matmul(mat1, mat2)
        arr4 = np.frombuffer(mat3, dtype=np.float64).reshape(mat3.getdims(), order='F', copy = False)
        flag = np.allclose(arr3, arr4)
        self.assertEqual(flag, True)

if __name__ == "__main__":
    unittest.main(verbosity=2)
