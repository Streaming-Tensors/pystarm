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
import pyttb as ttb
import os
import itertools

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

    def test_ttm_first_mode(self):
        """Test ttm in the first mode"""
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

        arr3 = np.frombuffer(ten2, dtype=np.float64).reshape(ten2.getdims(), order='F', copy = False)
        ten3 = ttb.tensor(arr1, copy=True).ttm(arr2, 0)
        flag = np.allclose(ten3.data, arr3)
        self.assertEqual(flag, True)

    def test_ttm_last_mode(self):
        """Test ttm in the last mode"""
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        mat_nelm = 3*2 # 6 elements
        mat_dims = (3, 2)
        mat_ndim = len(ten_dims)
        arr2 = np.arange(mat_nelm, dtype=np.float64).reshape(mat_dims, order='F')
        mat2 = pystarm.Matrix(arr2, mat_dims[0], mat_dims[1])
        
        ten2 = pystarm.ttm(ten1, mat2, 3) # On the last(4th) mode (3+1th mode)

        arr3 = np.frombuffer(ten2, dtype=np.float64).reshape(ten2.getdims(), order='F', copy = False)
        ten3 = ttb.tensor(arr1, copy=True).ttm(arr2, 3)
        flag = np.allclose(ten3.data, arr3)
        self.assertEqual(flag, True)

    def test_ttm_middle_mode(self):
        """Test ttm in a middle mode"""
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        mat_nelm = 2*3 # 6 elements
        mat_dims = (2, 3)
        mat_ndim = len(ten_dims)
        arr2 = np.arange(mat_nelm, dtype=np.float64).reshape(mat_dims, order='F')
        mat2 = pystarm.Matrix(arr2, mat_dims[0], mat_dims[1])
        
        ten2 = pystarm.ttm(ten1, mat2, 2) # On the third mode (2+1th mode)

        arr3 = np.frombuffer(ten2, dtype=np.float64).reshape(ten2.getdims(), order='F', copy = False)
        ten3 = ttb.tensor(arr1, copy=True).ttm(arr2, 2)
        flag = np.allclose(ten3.data, arr3)
        self.assertEqual(flag, True)

    def test_slicewise_svd(self):
        """Test slicewise SVD"""
        
        def slicewise_svd(A):
            dims    = A.shape
            nslices = np.prod(dims[2:])
            r       = min(A.shape[0], A.shape[1])
            U       = np.zeros((dims[0], r, nslices), dtype=np.float64, order='F')
            S       = np.zeros((r, nslices), dtype=np.float64, order='F')
            Vt      = np.zeros((r, dims[1], nslices), dtype=np.float64, order='F')

            slice_idx = 0
            # Order the frontal slices in natural mode ordering
            for idx in itertools.product(*reversed([range(p) for p in dims[2:]])):
              Aslice       = A[:, :, *reversed(idx)]
              [Us, s, Vts] = np.linalg.svd(Aslice, full_matrices=False)

              U[:, :, slice_idx]  = Us
              S[:, slice_idx]     = s
              Vt[:, :, slice_idx] = Vts

              slice_idx += 1

            return U, S, Vt
        
        def slicewise_checks(Apy, Aten):
            # Compute the slicewise SVD
            Uc, Sc, Vtc = pystarm.slicewise_svd(Aten)
            Spy2        = np.frombuffer(Sc, dtype=np.float64).reshape(Sc.getdims(), order='F', copy = False)

            # Compute in Python
            [Upy, Spy, Vtpy] = slicewise_svd(Apy)

            # Singular values should match exactly
            flag1 = np.allclose(Spy, Spy2)

            # Product of slices should be the same
            nslices = Spy.shape[1]
            sflags  = np.zeros(nslices, dtype=np.bool)

            for ii in range(nslices):
                Us   = Uc.getfrontalslice(ii)
                Usp  = np.frombuffer(Us, dtype=np.float64).reshape(Us.getdims(), order='F', copy=False)
                Vts  = Vtc.getfrontalslice(ii)
                Vtsp = np.frombuffer(Vts, dtype=np.float64).reshape(Vts.getdims(), order='F', copy=False)
                Ssp  = Spy2[:, ii]
                Bsp  = Usp @ (np.diag(Ssp) @ Vtsp)
        
                Bpy = Upy[:, :, ii] @ (np.diag(Spy[:, ii]) @ Vtpy[:, :, ii])

                # Check if the reconstructions are close
                sflags[ii] = np.allclose(Bsp, Bpy)

            flag2 = np.all(sflags)
              
            return flag1, flag2

        # Try two examples
        ## 3-dimensional example
        ten_nelm = 24
        ten_dims = (4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)
        
        [flag1, flag2] = slicewise_checks(arr1, ten1)
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)
      
        ## 4-dimensional example
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        [flag1, flag2] = slicewise_checks(arr1, ten1)
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)

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

    def test_svd(self):
        """Test matrix SVD"""
        Apy = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        Upy, spy, Vpyt = np.linalg.svd(Apy, full_matrices=False)

        Amat = pystarm.Matrix(Apy, 4, 3)
        Uc, sc, Vct = pystarm.svd(Amat)

        # Check if the arrays are not overwritten
        Apy2  = np.frombuffer(Amat, dtype=np.float64).reshape(Amat.getdims(), order='F', copy = False)
        flag1 = np.allclose(Apy, Apy2)
        self.assertEqual(flag1, True)

        # Check if the singular values are the same
        flag2 = np.allclose(sc, spy)
        self.assertEqual(flag2, True)

if __name__ == "__main__":
    unittest.main(verbosity=2)
