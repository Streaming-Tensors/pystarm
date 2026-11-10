import unittest
import numpy as np
import scipy.linalg
import pyttb as ttb
import os
import itertools
import gc
import time
import math
import sys
from alg import starM_product, starM_product_ttb, tensor_contraction_product_ttb, tensor_contract_multiply, tensor_contract_batched, contegra_tensor_contract_timed
import pystarm

# star-M-opt/ is not an importable package name (the hyphens are not valid in an
# identifier), so put the directory on sys.path and import the module directly.
_STAR_M_OPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'star-M-opt')
if _STAR_M_OPT_DIR not in sys.path:
    sys.path.insert(0, _STAR_M_OPT_DIR)
import svd_gradient_calculations as svdgrad


def np_slicewise_matmulks(U_jagged, S_jagged, Vt_jagged):
    T_slices = []
    nslices = len(U_jagged)
    for i in range(nslices):
        T_slices.append(np.matmul( np.matmul(U_jagged[i], np.diag(S_jagged[i]), order='F'), Vt_jagged[i], order='F') )
    T_np = np.stack(T_slices, axis=-1)
    return np.asfortranarray(T_np)


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

    def test_ttm_first_mode_transpose(self):
        """Test ttm in the first mode with a transposed matrix."""
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        mat_nelm = 3*5 # 15 elements
        mat_dims = (5, 3) # Make dimensions such that when it's transposed, they line up.
        mat_ndim = len(ten_dims)
        arr2 = np.arange(mat_nelm, dtype=np.float64).reshape(mat_dims, order='F')
        mat2 = pystarm.Matrix(arr2, mat_dims[0], mat_dims[1])

        ten2 = pystarm.ttm(ten1, mat2, 0, M_transpose=True) # On the first mode (0+1th mode)

        arr3 = np.frombuffer(ten2, dtype=np.float64).reshape(ten2.getdims(), order='F', copy = False)
        ten3 = ttb.tensor(arr1, copy=True).ttm(arr2.T, 0)
        flag = np.allclose(ten3.data, arr3)

        self.assertEqual(flag, True)

    def test_ttm_last_mode_transpose(self):
        """Test ttm in the last mode with a transposed matrix."""
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        mat_nelm = 3*2 # 6 elements
        mat_dims = (2, 3)
        mat_ndim = len(ten_dims)
        arr2 = np.arange(mat_nelm, dtype=np.float64).reshape(mat_dims, order='F')
        mat2 = pystarm.Matrix(arr2, mat_dims[0], mat_dims[1])

        ten2 = pystarm.ttm(ten1, mat2, 3, M_transpose=True) # On the last(4th) mode (3+1th mode)

        arr3 = np.frombuffer(ten2, dtype=np.float64).reshape(ten2.getdims(), order='F', copy = False)
        ten3 = ttb.tensor(arr1, copy=True).ttm(arr2.T, 3)
        flag = np.allclose(ten3.data, arr3)
        self.assertEqual(flag, True)

    def test_ttm_middle_mode_transpose(self):
        """Test ttm in a middle mode with a transposed matrix."""
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        mat_nelm = 2*3 # 6 elements
        mat_dims = (3, 2)
        mat_ndim = len(ten_dims)
        arr2 = np.arange(mat_nelm, dtype=np.float64).reshape(mat_dims, order='F')
        mat2 = pystarm.Matrix(arr2, mat_dims[0], mat_dims[1])

        ten2 = pystarm.ttm(ten1, mat2, 2, M_transpose=True) # On the third mode (2+1th mode)

        arr3 = np.frombuffer(ten2, dtype=np.float64).reshape(ten2.getdims(), order='F', copy = False)
        ten3 = ttb.tensor(arr1, copy=True).ttm(arr2.T, 2)
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

            Atilde_c = pystarm.slicewise_matmul(Uc, Sc, Vtc)
            Atilde_np = np.frombuffer(Atilde_c, dtype=np.float64).reshape(Atilde_c.getdims(), order='F', copy = False)
            # Atilde_py = np.zeros((Upy.shape[0], Vtpy.shape[1], nslices))
            Atilde_py = np.zeros(Atilde_np.shape)

            for ii in range(nslices):
                Us   = Uc.getfrontalslice(ii)
                Usp  = np.frombuffer(Us, dtype=np.float64).reshape(Us.getdims(), order='F', copy=False)
                Vts  = Vtc.getfrontalslice(ii)
                Vtsp = np.frombuffer(Vts, dtype=np.float64).reshape(Vts.getdims(), order='F', copy=False)
                Ssp  = Spy2[:, ii]
                Bsp  = Usp @ (np.diag(Ssp) @ Vtsp)
                Bpy = Upy[:, :, ii] @ (np.diag(Spy[:, ii]) @ Vtpy[:, :, ii])
                Atilde_py[:, :, ii] = Bpy

                # Check if the reconstructions are close
                sflags[ii] = np.allclose(Bsp, Bpy)

            flag2 = np.all(sflags)

            Atilde_py = Atilde_py.reshape(Atilde_np.shape)
            flag3 = np.allclose(Atilde_np, Atilde_py)
            # print("Atilde_py", Atilde_py)
            # print("Atilde_np", Atilde_np)
            # print("Error", np.linalg.norm(np.abs(Atilde_py - Atilde_np)))
              
            return flag1, flag2, flag3

        def slicewise_checks_US_only(Apy, Aten):
           # Checking on Tensor Tensor-Diagonal product with random tensors, as checking with U and S values
           # Will produce different results as numpy and pystarm calculate U values differently. 
            ten_nelm = 24
            ten_dims = (4, 3, 2)
            ten_ndim = len(ten_dims)
            Apy = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
            Aten = pystarm.Tensor(Apy, ten_ndim, ten_dims)

            n_slices = np.prod(ten_dims[2:])
            M_dims = (ten_dims[1], n_slices)
            M_nelm = np.prod(M_dims)
            Mpy = np.arange(M_nelm, dtype=np.float64).reshape(M_dims, order='F')
            M = pystarm.Matrix(Mpy, M_dims[0], M_dims[1])

            Atilde_c = pystarm.slicewise_matmul(Aten, M)
            Atilde_np = np.frombuffer(Atilde_c, dtype=np.float64).reshape(Atilde_c.getdims(), order='F', copy = False)
            Atilde_py = np.zeros(Atilde_np.shape)

            for ii in range(n_slices):
                Bpy = Apy[:, :, ii] @ np.diag(Mpy[:, ii])
                Atilde_py[:, :, ii] = Bpy


            Atilde_py = Atilde_py.reshape(Atilde_np.shape)
            return np.allclose(Atilde_np, Atilde_py)

        # Try two examples
        ## 3-dimensional example
        ten_nelm = 24
        ten_dims = (4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)
        
        [flag1, flag2, flag3] = slicewise_checks(arr1, ten1)
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)
        self.assertEqual(flag3, True)

        # Check for just U @ S
        flag1 = slicewise_checks_US_only(arr1, ten1)
        self.assertEqual(flag1, True)

        # ## 4-dimensional example
        # ten_nelm = 120
        # ten_dims = (5, 4, 3, 2)
        # ten_ndim = len(ten_dims)
        # arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        # ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        # [flag1, flag2, flag3] = slicewise_checks(arr1, ten1)
        # self.assertEqual(flag1, True)
        # self.assertEqual(flag2, True)
        # self.assertEqual(flag3, True)

    def test_slicewise_svdx(self):
        """Test slicewise truncated SVD"""
        
        def slicewise_svdx(A, k):
            dims    = A.shape
            nslices = np.prod(dims[2:])
            U       = np.zeros((dims[0], k, nslices), dtype=np.float64, order='F')
            S       = np.zeros((k, nslices), dtype=np.float64, order='F')
            Vt      = np.zeros((k, dims[1], nslices), dtype=np.float64, order='F')

            slice_idx = 0
            # Order the frontal slices in natural mode ordering
            for idx in itertools.product(*reversed([range(p) for p in dims[2:]])):
              Aslice       = A[:, :, *reversed(idx)]
              [Us, s, Vts] = np.linalg.svd(Aslice, full_matrices=False)

              U[:, :, slice_idx]  = Us[:, :k]
              S[:, slice_idx]     = s[:k]
              Vt[:, :, slice_idx] = Vts[:k, :]

              slice_idx += 1

            return U, S, Vt
        
        def slicewise_checks(Apy, Aten, k):
            # Compute the slicewise SVD
            Uc, Sc, Vtc = pystarm.slicewise_svdx(Aten, k)
            Spy2        = np.frombuffer(Sc, dtype=np.float64).reshape(Sc.getdims(), order='F', copy = False)

            # Compute in Python
            [Upy, Spy, Vtpy] = slicewise_svdx(Apy, k)

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
        k = 2

        ## 3-dimensional example
        ten_nelm = 24
        ten_dims = (4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)
        
        [flag1, flag2] = slicewise_checks(arr1, ten1, k)
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)
      
        ## 4-dimensional example
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        [flag1, flag2] = slicewise_checks(arr1, ten1, k)
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)

    def test_slicewise_svdks(self):
        """Test slicewise truncated SVD with different ranks per slice"""
        
        def slicewise_svdks(A, ks):
            dims    = A.shape
            nslices = np.prod(dims[2:])
            U       = []
            S       = []
            Vt      = []

            slice_idx = 0
            # Order the frontal slices in natural mode ordering
            for idx in itertools.product(*reversed([range(p) for p in dims[2:]])):
              Aslice       = A[:, :, *reversed(idx)]
              [Us, s, Vts] = np.linalg.svd(Aslice, full_matrices=False)

              k = ks[slice_idx]
              U.append(Us[:, :k])
              S.append(s[:k])
              Vt.append(Vts[:k, :])

              slice_idx += 1

            return U, S, Vt
        
        def slicewise_checks(Apy, Aten, ks):
            # Compute the slicewise SVD
            Uc, Sc, Vtc = pystarm.slicewise_svdks(Aten, ks)

            # Compute in Python
            [Upy, Spy, Vtpy] = slicewise_svdks(Apy, ks)

            # Product of slices should be the same
            nslices = len(Spy)
            sflags  = np.zeros(nslices, dtype=np.bool)

            for ii in range(nslices):
                Us   = Uc.getfrontalslice(ii)
                Usp  = np.frombuffer(Us, dtype=np.float64).reshape(Us.getdims(), order='F', copy=False)
                Vts  = Vtc.getfrontalslice(ii)
                Vtsp = np.frombuffer(Vts, dtype=np.float64).reshape(Vts.getdims(), order='F', copy=False)
                Ssp  = Sc.getcol(ii)
                Bsp  = Usp @ (np.diag(Ssp) @ Vtsp)
        
                Bpy = Upy[ii] @ (np.diag(Spy[ii]) @ Vtpy[ii])

                # Check if the reconstructions are close
                sflags[ii] = np.allclose(Bsp, Bpy)

            flag = np.all(sflags)
              
            return flag

        # Try three examples

        ## 3-dimensional example
        ks       = [2, 1]
        ten_nelm = 24
        ten_dims = (4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)
        
        flag = slicewise_checks(arr1, ten1, ks)
        self.assertEqual(flag, True)
      
        ## 4-dimensional example
        ks       = [4, 3, 1, 2, 2, 3]
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        flag = slicewise_checks(arr1, ten1, ks)
        self.assertEqual(flag, True)

        ## 4-dimensional example with zeros
        ks       = [4, 3, 0, 2, 0, 3]
        ten_nelm = 120
        ten_dims = (5, 4, 3, 2)
        ten_ndim = len(ten_dims)
        arr1 = np.arange(ten_nelm, dtype=np.float64).reshape(ten_dims, order='F')
        ten1 = pystarm.Tensor(arr1, ten_ndim, ten_dims)

        flag = slicewise_checks(arr1, ten1, ks)
        self.assertEqual(flag, True)

    def test_slicewise_svd_thr(self):
        """Test slicewise truncated SVD given an energy threshold while saving intermediate computations"""

        def create_range_tensors(dims):
          A = np.arange(np.prod(dims), dtype=np.float64).reshape(dims, order='F')
          return A

        def check_slicewise_svd_thr(dims, thr):
          Ap = create_range_tensors(dims)
          Ac = pystarm.Tensor(Ap, len(Ap.shape), Ap.shape)

          Sf = pystarm.slicewise_svdvals(Ac)
          ks = pystarm.thresholds(Sf, thr)

          U1, S1, V1t = pystarm.slicewise_svdks(Ac, ks)
          U2, S2, V2t = pystarm.slicewise_svd_thr(Ac, thr)

          nslices = np.prod(dims[2:])
          Uflags  = np.zeros(nslices, dtype=np.bool)
          Sflags  = np.zeros(nslices, dtype=np.bool)
          Vtflags = np.zeros(nslices, dtype=np.bool)

          for ii in range(nslices):
            U1s  = U1.getfrontalslice(ii)
            U1sp = np.frombuffer(U1s, dtype=np.float64).reshape(U1s.getdims(), order='F', copy=False)
            U2s  = U2.getfrontalslice(ii)
            U2sp = np.frombuffer(U2s, dtype=np.float64).reshape(U2s.getdims(), order='F', copy=False)
            Uflags[ii] = np.allclose(U1sp, U2sp)

            V1ts  = V1t.getfrontalslice(ii)
            V1tsp = np.frombuffer(V1ts, dtype=np.float64).reshape(V1ts.getdims(), order='F', copy=False)
            V2ts  = V2t.getfrontalslice(ii)
            V2tsp = np.frombuffer(V2ts, dtype=np.float64).reshape(V2ts.getdims(), order='F', copy=False)
            Vtflags[ii] = np.allclose(V1tsp, V2tsp)

            Sflags[ii]  = np.allclose(S1.getcol(ii), S2.getcol(ii))

          return Uflags, Sflags, Vtflags

        # Try all paths

        all_dims = [(10, 4, 3, 2), (5, 4, 3, 2), (3, 4, 3, 2), (3, 10, 3, 2)]
        thrs     = [0.3, 0.001]

        for dims in all_dims:
          for thr in thrs:
            uflags, sflags, vtflags = check_slicewise_svd_thr(dims, thr)
            self.assertEqual(np.all(uflags), True)
            self.assertEqual(np.all(sflags), True)
            self.assertEqual(np.all(vtflags), True)

    def test_slicewise_matmulks_fixed_case(self):
        """Test slicewise matmul with different ranks per slice with fixed test case"""
        U_fixed_dim_size = 5
        U_nslices = 4
        U_slice_ranks = [2,3,2,2]
        U_jagged = []
        U_flattened_list = []
        for i in range(U_nslices):
            slice_vals = i+1 
            x = slice_vals * np.ones((U_fixed_dim_size, U_slice_ranks[i]), dtype=np.float64, order='F')
            U_jagged.append(x)
            U_flattened_list.append(x.flatten(order='F'))
        U_flattened = np.concatenate(U_flattened_list)
        U_starm = pystarm.JaggedTensor(U_flattened, U_fixed_dim_size, U_slice_ranks)

        S_ncol = U_nslices
        S_col_ranks = U_slice_ranks
        S_jagged = []
        S_flattened_list = []
        for i in range(S_ncol):
            val = i+1
            x = val * np.ones(S_col_ranks[i], dtype=np.float64, order='F')
            S_jagged.append(x)
            S_flattened_list.append(x.flatten(order='F'))
        S_flattened = np.concatenate(S_flattened_list)
        S_starm = pystarm.JaggedMatrix(S_flattened, S_col_ranks)

        Vt_fixed_dim_size = 3
        Vt_nslices = U_nslices
        Vt_slice_ranks = U_slice_ranks
        Vt_jagged = []
        Vt_flattened_list = []
        for i in range(Vt_nslices):
            slice_vals = i+1
            x = slice_vals * np.ones((Vt_slice_ranks[i], Vt_fixed_dim_size), dtype=np.float64, order='F')
            Vt_jagged.append(x)
            Vt_flattened_list.append(x.flatten(order='F'))
        Vt_flattened = np.concatenate(Vt_flattened_list)
        Vt_starm = pystarm.JaggedTensor(Vt_flattened, Vt_fixed_dim_size, Vt_slice_ranks, True)

        T_starm = pystarm.slicewise_matmulks(U_starm, S_starm, Vt_starm) 
        T_py  = np.frombuffer(T_starm, dtype=np.float64).reshape(T_starm.getdims(), order='F', copy=False)
        T_np = np_slicewise_matmulks(U_jagged, S_jagged, Vt_jagged)

        flag = np.allclose(T_py.flatten(order='F'), T_np.flatten(order='F'))
        self.assertEqual(flag, True)

    def test_slicewise_matmulks_random_case(self):
        """Test slicewise matmul with different ranks per slice with random test case"""
        nslices = np.random.randint(1, 20)
        slice_ranks = np.random.randint(1, 10, nslices)

        U_fixed_dim_size = np.random.randint(1, 20)
        U_nslices = nslices
        U_slice_ranks = slice_ranks
        U_jagged = []
        U_flattened_list = []
        for i in range(U_nslices):
            x = np.asfortranarray(np.random.rand(U_fixed_dim_size, U_slice_ranks[i]).astype(np.float64))
            U_jagged.append(x)
            U_flattened_list.append(x.flatten(order='F'))
        U_flattened = np.concatenate(U_flattened_list)
        U_starm = pystarm.JaggedTensor(U_flattened, U_fixed_dim_size, U_slice_ranks)

        S_ncol = U_nslices
        S_col_ranks = U_slice_ranks
        S_jagged = []
        S_flattened_list = []
        for i in range(S_ncol):
            # val = i+1
            # x = val * np.ones(S_col_ranks[i], dtype=np.float64, order='F')
            x = np.asfortranarray(np.random.rand(S_col_ranks[i]).astype(np.float64))
            S_jagged.append(x)
            S_flattened_list.append(x.flatten(order='F'))
        S_flattened = np.concatenate(S_flattened_list)
        S_starm = pystarm.JaggedMatrix(S_flattened, S_col_ranks)

        Vt_fixed_dim_size = np.random.randint(1, 20)
        Vt_nslices = U_nslices
        Vt_slice_ranks = U_slice_ranks
        Vt_jagged = []
        Vt_flattened_list = []
        for i in range(Vt_nslices):
            # slice_vals = i+1
            # x = slice_vals * np.ones((Vt_slice_ranks[i], Vt_fixed_dim_size), dtype=np.float64, order='F')
            x = np.asfortranarray(np.random.rand(Vt_slice_ranks[i], Vt_fixed_dim_size).astype(np.float64))
            Vt_jagged.append(x)
            Vt_flattened_list.append(x.flatten(order='F'))
        Vt_flattened = np.concatenate(Vt_flattened_list)
        Vt_starm = pystarm.JaggedTensor(Vt_flattened, Vt_fixed_dim_size, Vt_slice_ranks, True)

        T_starm = pystarm.slicewise_matmulks(U_starm, S_starm, Vt_starm) 
        T_py  = np.frombuffer(T_starm, dtype=np.float64).reshape(T_starm.getdims(), order='F', copy=False)
        T_np = np_slicewise_matmulks(U_jagged, S_jagged, Vt_jagged)

        flag = np.allclose(T_py.flatten(order='F'), T_np.flatten(order='F'))
        self.assertEqual(flag, True)

    def test_slicewise_matmulks_random_case_with_empty_slices(self):
        """Test slicewise matmul with different ranks per slice with random test case with empty slices"""
        nslices = 6
        slice_ranks = np.array([4,3,0,2,0,3])

        # U_fixed_dim_size = np.random.randint(1, 20)
        U_fixed_dim_size = 5
        U_nslices = nslices
        U_slice_ranks = slice_ranks
        U_jagged = []
        U_flattened_list = []
        for i in range(U_nslices):
            slice_vals = i+1
            x = slice_vals * np.ones((U_fixed_dim_size, U_slice_ranks[i]), dtype=np.float64, order='F')
            # x = np.asfortranarray(np.random.rand(U_fixed_dim_size, U_slice_ranks[i]).astype(np.float64))
            U_jagged.append(x)
            U_flattened_list.append(x.flatten(order='F'))
            # print('U', i, U_flattened_list[-1])
        U_flattened = np.concatenate(U_flattened_list)
        U_starm = pystarm.JaggedTensor(U_flattened, U_fixed_dim_size, U_slice_ranks)

        S_ncol = U_nslices
        S_col_ranks = U_slice_ranks
        S_jagged = []
        S_flattened_list = []
        for i in range(S_ncol):
            val = i+1
            x = val * np.ones(S_col_ranks[i], dtype=np.float64, order='F')
            # x = np.asfortranarray(np.random.rand(S_col_ranks[i]).astype(np.float64))
            S_jagged.append(x)
            S_flattened_list.append(x.flatten(order='F'))
            # print('S', i, S_flattened_list[-1])
        S_flattened = np.concatenate(S_flattened_list)
        S_starm = pystarm.JaggedMatrix(S_flattened, S_col_ranks)

        # Vt_fixed_dim_size = np.random.randint(1, 20)
        Vt_fixed_dim_size = 3
        Vt_nslices = U_nslices
        Vt_slice_ranks = U_slice_ranks
        Vt_jagged = []
        Vt_flattened_list = []
        for i in range(Vt_nslices):
            slice_vals = i+1
            x = slice_vals * np.ones((Vt_slice_ranks[i], Vt_fixed_dim_size), dtype=np.float64, order='F')
            # x = np.asfortranarray(np.random.rand(Vt_slice_ranks[i], Vt_fixed_dim_size).astype(np.float64))
            Vt_jagged.append(x)
            Vt_flattened_list.append(x.flatten(order='F'))
            # print('Vt', i, Vt_flattened_list[-1])
        Vt_flattened = np.concatenate(Vt_flattened_list)
        # print('U', U_flattened)
        # print('S', S_flattened)
        # print('Vt', Vt_flattened)
        # print('slice_ranks', Vt_slice_ranks)
        Vt_starm = pystarm.JaggedTensor(Vt_flattened, Vt_fixed_dim_size, Vt_slice_ranks, True)

        T_starm = pystarm.slicewise_matmulks(U_starm, S_starm, Vt_starm) 
        T_py  = np.frombuffer(T_starm, dtype=np.float64).reshape(T_starm.getdims(), order='F', copy=False)
        T_np = np_slicewise_matmulks(U_jagged, S_jagged, Vt_jagged)

        # print(T_py.flatten(order='F'))
        # print(T_np.flatten(order='F'))

        flag = np.allclose(T_py.flatten(order='F'), T_np.flatten(order='F'))
        self.assertEqual(flag, True)

    def test_truncate_factors(self):
        """Test truncating full tSVDM factors to rank-k"""
        def create_full_tensors(dims):
          U_dims    = list(dims)
          U_dims[1] = U_dims[0]
          U_dims    = tuple(U_dims)
          U = np.arange(np.prod(U_dims), dtype=np.float64).reshape(U_dims, order='F')

          Vt_dims    = list(dims)
          Vt_dims[0] = Vt_dims[1]
          Vt_dims    = tuple(Vt_dims)
          Vt = np.arange(np.prod(Vt_dims), dtype=np.float64).reshape(Vt_dims, order='F')

          r = min(dims[0], dims[1])
          s = np.prod(dims[2:])
          S = np.arange(r*s, dtype=np.float64).reshape((r,s), order='F')

          return U, S, Vt

        def truncate_tensor(A, ranks, jaggedfirstmode=False):
          dims    = A.shape
          nslices = np.prod(dims[2:])

          # Store truncated slices in a list
          Aslices = []
          sidx    = 0
          # Order the frontal slices in natural mode ordering
          for idx in itertools.product(*reversed([range(p) for p in dims[2:]])):
            Aslice = A[:, :, *reversed(idx)].copy()
            if (jaggedfirstmode): # Forming Vt slices
              Aslices.append(Aslice[:ranks[sidx], :])
            else: # Forming U slices
              Aslices.append(Aslice[:, :ranks[sidx]])
            sidx += 1

          return Aslices

        def truncate_matrix(M, ranks):
          Mslices = []
          for j in range(M.shape[1]):
            Mslices.append(M[:ranks[j], j])

          return Mslices

        def check_truncation(dims, ranks):
          Up, Sp, Vtp = create_full_tensors(dims)

          Uc  = pystarm.Tensor(Up, len(Up.shape), Up.shape)
          Sc  = pystarm.Matrix(Sp, Sp.shape[0], Sp.shape[1])
          Vtc = pystarm.Tensor(Vtp, len(Vtp.shape), Vtp.shape)

          Upk  = truncate_tensor(Up, ranks)
          Spk  = truncate_matrix(Sp, ranks)
          Vtpk = truncate_tensor(Vtp, ranks, True)

          Uck, Sck, Vtck = pystarm.truncate_factors(Uc, Sc, Vtc, ranks)

          nslices = np.prod(dims[2:])
          Uflags  = np.zeros(nslices, dtype=np.bool)
          Sflags  = np.zeros(nslices, dtype=np.bool)
          Vtflags = np.zeros(nslices, dtype=np.bool)

          for ii in range(nslices):
            Us  = Uck.getfrontalslice(ii)
            Usp = np.frombuffer(Us, dtype=np.float64).reshape(Us.getdims(), order='F', copy=False)
            Uflags[ii] = np.allclose(Usp, Upk[ii])

            Vts  = Vtck.getfrontalslice(ii)
            Vtsp = np.frombuffer(Vts, dtype=np.float64).reshape(Vts.getdims(), order='F', copy=False)
            Vtflags[ii] = np.allclose(Vtsp, Vtpk[ii])

            Sflags[ii]  = np.allclose(Sck.getcol(ii), Spk[ii])

          return Uflags, Sflags, Vtflags

        # Try three examples

        ## 3-dimensional example
        ks       = [2, 1]
        ten_dims = (4, 3, 2)

        uflags, sflags, vtflags = check_truncation(ten_dims, ks)

        self.assertEqual(np.all(uflags), True)
        self.assertEqual(np.all(sflags), True)
        self.assertEqual(np.all(vtflags), True)

        ## 4-dimensional example
        ks       = [4, 3, 1, 2, 2, 3]
        ten_dims = (5, 4, 3, 2)

        uflags, sflags, vtflags = check_truncation(ten_dims, ks)

        self.assertEqual(np.all(uflags), True)
        self.assertEqual(np.all(sflags), True)
        self.assertEqual(np.all(vtflags), True)

        ## 4-dimensional example with zeros
        ks       = [4, 3, 0, 2, 0, 3]
        ten_dims = (5, 4, 3, 2)

        uflags, sflags, vtflags = check_truncation(ten_dims, ks)

        self.assertEqual(np.all(uflags), True)
        self.assertEqual(np.all(sflags), True)
        self.assertEqual(np.all(vtflags), True)


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

    def test_matrix_gc_behavior(self):
        """Test if garbage collection of the original numpy object changes the underlying buffer of pystarm matrix"""

        arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        mat1 = pystarm.Matrix(arr1, 4, 3)
        
        del arr1
        gc.collect()

        arr3 = np.ones(12, dtype=np.float64)

        arr4 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        arr2 = np.frombuffer(mat1, dtype=np.float64).reshape(mat1.getdims(), order='F', copy = False)
        flag = np.allclose(arr4, arr2)
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

    def test_svdvals(self):
        """Test matrix SVD (singular values only)"""
        Apy = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        _, spy, _ = np.linalg.svd(Apy, full_matrices=False)

        Amat = pystarm.Matrix(Apy, 4, 3)
        sc   = pystarm.svdvals(Amat)

        # Check if the arrays are not overwritten
        Apy2  = np.frombuffer(Amat, dtype=np.float64).reshape(Amat.getdims(), order='F', copy = False)
        flag1 = np.allclose(Apy, Apy2)
        self.assertEqual(flag1, True)

        # Check if the singular values are the same
        flag2 = np.allclose(sc, spy)
        self.assertEqual(flag2, True)

    def test_svdx(self):
        """Test matrix truncated SVD"""
        Apy = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
        Upy, spy, Vpyt = np.linalg.svd(Apy, full_matrices=False)

        # Truncate to rank two
        k = 2
        Upy  = Upy[:, :k]
        spy  = spy[:k]
        Vpyt = Vpyt[:k, :]

        Amat = pystarm.Matrix(Apy, 4, 3)
        Uc, sc, Vct = pystarm.svdx(Amat, k)

        # Check if the arrays are not overwritten
        Apy2  = np.frombuffer(Amat, dtype=np.float64).reshape(Amat.getdims(), order='F', copy = False)
        flag1 = np.allclose(Apy, Apy2)
        self.assertEqual(flag1, True)

        # Check if the singular values are the same
        flag2 = np.allclose(sc, spy)
        self.assertEqual(flag2, True)

    def test_svdvals_threshold(self):
        """Test finding the threshold singular value"""
        Apy  = np.random.rand(12).reshape((4,3), order='F')

        for j in range(Apy.shape[1]):
          b        = np.sort(Apy[:,j])
          Apy[:,j] = b[::-1]

        Amat = pystarm.Matrix(Apy, 4, 3)

        # Run the test for the following thresholds
        tols  = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 0.7]
        flags = np.zeros(len(tols))
        a     = Apy.flatten().copy()
        a     = np.sort(a)
        b     = a**2
        b     = np.cumsum(b) / np.sum(b)

        for i,t in enumerate(tols):
          t_vals = np.where(b < t**2)[0] 
          if (len(t_vals) > 0):
            t1 = a[t_vals[-1]] # Get highest threshold
          else:
            t1 = 0.0
          t2       = pystarm.threshold(Amat, t)
          flags[i] = np.allclose(t1, t2)
        
        self.assertEqual(np.all(flags), True)
          
    def test_columnwise_ranks(self):
        """Test finding the columnwise ranks given a threshold"""
        Apy  = np.random.rand(12).reshape((4,3), order='F')

        for j in range(Apy.shape[1]):
          b        = np.sort(Apy[:,j])
          Apy[:,j] = b[::-1]

        Amat = pystarm.Matrix(Apy, 4, 3)

        # Get the singular values sorted
        a     = Apy.flatten().copy()
        a     = np.sort(a)
        b     = a**2
        b     = np.cumsum(b) / np.sum(b)

        # Run the test for the following thresholds
        tols  = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 0.7]
        flags = np.zeros(len(tols))

        for i,t in enumerate(tols):
          col_ranks = np.zeros(Apy.shape[1])

          t_vals = np.where(b < t**2)[0] 
          if (len(t_vals) > 0):
            thr = a[t_vals[-1]] # Get highest threshold
          else:
            thr = 0.0

          for j in range(Apy.shape[1]):
            col_ranks[j] = np.sum(Apy[:,j] >= thr)
                
          col_ranks2 = pystarm.thresholds(Amat, t)
          flags[i]   = np.allclose(col_ranks, col_ranks2)
        
        self.assertEqual(np.all(flags), True)

class JaggedTensorTestCase(unittest.TestCase):
    def test_jagged_tensor_creation(self):
        """Test jagged tensor creation and operations"""
        # Create a python jagged tensor
        fixed_dim = np.random.randint(1, 20)
        # print("fixed_dim", fixed_dim)
        nslices = np.random.randint(1, 20)
        # print("nslices", nslices)
        slice_ranks = np.random.randint(1, 10, nslices)
        # print("slice_ranks", slice_ranks)
        nelements = sum(fixed_dim * slice_ranks)
        # print("nelements", nelements)
        vals = np.random.rand(nelements)
        jagged_ten = pystarm.JaggedTensor(vals, fixed_dim, slice_ranks)
        jagged_ten_vals = np.frombuffer(jagged_ten, dtype=np.float64)
        flag = np.allclose(vals, jagged_ten_vals)
        self.assertEqual(flag, True)

    def test_jagged_tensor_getslice(self):
        """Test jagged tensor get slice"""
        # Create a python jagged tensor
        fixed_dim = np.random.randint(1, 20)
        nslices = np.random.randint(1, 20)
        slice_ranks = np.random.randint(1, 10, nslices)
        nelements = sum(fixed_dim * slice_ranks)
        vals = np.random.rand(nelements)
        jagged_ten = pystarm.JaggedTensor(vals, fixed_dim, slice_ranks)
        # compute nrows for each slice
        # get each slice and check again python vals
        # need to use offsets since vals is a 1D array
        start_idx = 0
        end_idx = 0
        for i in range(nslices):
            end_idx += fixed_dim * slice_ranks[i]
            slice = jagged_ten.getfrontalslice(i)
            slice_vals = np.frombuffer(slice, dtype=np.float64)
            py_slice_vals = vals[start_idx:end_idx]
            flag = np.allclose(slice_vals, py_slice_vals)
            start_idx = end_idx
            self.assertEqual(flag, True)

    def test_jagged_tensor_setslice(self):
        """Test jagged tensor set slice"""
        # Create a python jagged tensor
        fixed_dim = np.random.randint(1, 20)
        nslices = np.random.randint(1, 20)
        slice_ranks = np.random.randint(1, 10, nslices)
        nelements = sum(fixed_dim * slice_ranks)
        vals = np.random.rand(nelements)
        jagged_ten = pystarm.JaggedTensor(vals, fixed_dim, slice_ranks)
        # compute nrows for each slice
        # get each slice and check again python vals
        # need to use offsets since vals is a 1D array
        start_idx = 0
        end_idx = 0
        new_vals = np.zeros_like(vals)
        for i in range(nslices):
            end_idx += fixed_dim * slice_ranks[i]

            mat_nelm = fixed_dim*slice_ranks[i]
            mat_dims = (fixed_dim, slice_ranks[i])
            mat_ndim = len(mat_dims)
            arr = np.arange(mat_nelm, dtype=np.float64).reshape(mat_dims, order='F')
            # print("Setting slice:", i)
            # print(arr)
            new_vals[start_idx:end_idx] = arr.flatten(order='F')
            # print(new_vals[start_idx:end_idx])
            mat = pystarm.Matrix(arr, mat_dims[0], mat_dims[1])
            jagged_ten.setfrontalslice(mat, i)
            start_idx = end_idx

        jagged_ten_vals = np.frombuffer(jagged_ten, dtype=np.float64)
        flag = np.allclose(jagged_ten_vals, new_vals)
        # print("jagged_ten_vals", jagged_ten_vals)
        # print("new_vals", new_vals)
        self.assertEqual(flag, True)

class JaggedMatrixTestCase(unittest.TestCase):
    def test_jagged_matrix_creation(self):
        """Test jagged matrix creation and operations"""
        # print("fixed_dim", fixed_dim)
        ncol = 4
        col_ranks = [2,3,2,2] 
        nelements = 9 
        vals = np.random.rand(nelements)
        jagged_mat = pystarm.JaggedMatrix(vals, col_ranks)
        jagged_mat_vals = np.frombuffer(jagged_mat, dtype=np.float64)
        flag = np.allclose(vals, jagged_mat_vals)
        self.assertEqual(flag, True)


class StarMProductTestCase(unittest.TestCase):
    '''
    This tests the functionality of the starM product algorithm outlined by the
    `starM_product` function in `alg.py`
    '''
    def test_third_order_product(self):
       '''Test a third order starM product'''
       # We will say that A has dimensions m x p x n_slices
       # B will have dimensions p x l x n_slices
       # M will be an identity matrix with dimensions n_slices x n_slices
       
       # Keep all dimensions between 2 and 10 to make sure it doesn't run too long.
       n_slices = np.random.randint(2, 10)
       m = np.random.randint(2,10)
       p = np.random.randint(2,10)
       l = np.random.randint(2,10)

       A_nelm = m * p * n_slices
       A_dims = (m, p, n_slices)

       B_nelm = p * l * n_slices
       B_dims = (p, l, n_slices)
       
       A = np.random.rand(A_nelm).reshape(A_dims, order='F')
       B = np.random.rand(B_nelm).reshape(B_dims, order='F')
       M = np.identity(n_slices)

       C_pystarm = starM_product(A,B,M)
       C_ttb = starM_product_ttb(A,B,M)

       self.assertTrue(np.allclose(C_pystarm, C_ttb))
    
    def test_fourth_order_product(self):
        '''Test a fourth-order starM product.'''
        n = np.random.randint(2,10)
        m = np.random.randint(2,10)
        p = np.random.randint(2,10)
        l = np.random.randint(2,10)
        q = np.random.randint(2,10)

        A_nelm = m * p * n * q
        A_dims = (m, p, n, q)

        B_nelm = p * l * n * q
        B_dims = (p, l, n, q)
        
        A = np.random.rand(A_nelm).reshape(A_dims, order='F')
        B = np.random.rand(B_nelm).reshape(B_dims, order='F')
        M = [np.identity(n) for n in A_dims[2:]]

        C_pystarm = starM_product(A,B,M)
        C_ttb = starM_product_ttb(A,B,M)

        self.assertTrue(np.allclose(C_pystarm, C_ttb))

    # TODO: Fix this test case. See starM_product in alg.py
    # def test_third_order_product_transpose(self):
    #    '''Test a third order starM product while transposing both slices'''
    #    # We will say that A has dimensions m x p x n_slices
    #    # B will have dimensions p x l x n_slices
    #    # M will be an identity matrix with dimensions n_slices x n_slices
       
    #    # Keep all dimensions between 2 and 10 to make sure it doesn't run too long.
    #    n_slices = np.random.randint(2, 10)
    #    m = np.random.randint(2,10)
    #    p = np.random.randint(2,10)
    #    l = np.random.randint(2,10)

    #    A_nelm = m * p * n_slices
    #    A_dims = (m, p, n_slices)

    #    B_nelm = p * l * n_slices
    #    B_dims = (p, l, n_slices)
       
    #    A = np.random.rand(A_nelm).reshape(A_dims, order='F')
    #    B = np.random.rand(B_nelm).reshape(B_dims, order='F')
    #    M = np.identity(n_slices)

    #    C_w_transpose = starM_product(B,A,M, A_transpose=True, B_transpose=True)
    #    C_wo_transpose = starM_product(A,B,M,A_transpose=False, B_transpose=False)

    #    self.assertFalse(np.allclose(C_w_transpose, C_wo_transpose.T))

class ContractionTestCase(unittest.TestCase):
    # TODO: Free the tensors and ensure no memory leaks. 
    def test_order_three_contraction(self):
        '''
        Test a tensor contraction with all modes of two mode-3 tensors. Testing batched implementation, OMP reduction, and cotengra implementation.
        '''
        modes = [0, 1, 2]
        flags = []
        for mode in modes:
            n_k = np.random.randint(2, 10)
            p = np.random.randint(2,10)
            m = np.random.randint(2,10)

            # For testing integrity, make sure all dims except n_k and p are equal.
            A_nelm = n_k * m * m
            A_dims = [m, m, m]
            A_dims[mode] = n_k

            B_nelm =  p * m * m
            B_dims = [m, m, m]
            B_dims[mode] = p

            A = np.random.rand(A_nelm).reshape(A_dims, order='F')
            B = np.random.rand(B_nelm).reshape(B_dims, order='F')
            
            C_pystarm = tensor_contract_multiply(A, B, mode, naive=False)
            C_pystarm_batched, _ = tensor_contract_batched(A, B, mode)
            C_ttb = tensor_contraction_product_ttb(A, B, mode)
            C_cotengra, _ = contegra_tensor_contract_timed(A, B, mode)
            flags.append(np.allclose(C_pystarm, C_ttb, C_pystarm_batched, C_cotengra))

    def test_order_four_contractions(self):
        '''
        Test a tensor contractions with all modes of two mode-4 tensors. Testing both the batched, OMP, and cotengra reduction cases.
        '''
        modes = [0, 1, 2, 3]
        flags = []
        n_k = np.random.randint(2, 10)
        p = np.random.randint(2,10)
        m = np.random.randint(2,10)
        for mode in modes:
            
            # For testing integrity, make sure all dims except n_k and p are equal.
            A_nelm = n_k * (m ** (len(modes) - 1))
            A_dims = [m for i in modes]
            A_dims[mode] = n_k

            B_nelm =  p *  (m ** (len(modes) - 1))
            B_dims = [m for i in modes]
            B_dims[mode] = p

            A = np.random.rand(A_nelm).reshape(A_dims, order='F')
            B = np.random.rand(B_nelm).reshape(B_dims, order='F')
            
            C_pystarm = tensor_contract_multiply(A, B, mode, naive=False)
            C_pystarm_batched, _ = tensor_contract_batched(A, B, mode)
            C_ttb = tensor_contraction_product_ttb(A, B, mode)
            C_cotengra, _ = contegra_tensor_contract_timed(A, B, mode)
            flags.append(np.allclose(C_pystarm, C_ttb, C_pystarm_batched, C_cotengra))

        self.assertEqual(np.all(flags), True)

class TensorArithmeticTestCase(unittest.TestCase):
    def create_rand_np_tensors(self):
        n = 5
        A_dims = (n,n,n,n)
        A_nelm = math.prod(A_dims)

        B_dims = (n,n,n,n)
        B_nelm = math.prod(B_dims)

        # Create numpy tensors
        A_np = np.asfortranarray(np.random.rand(A_nelm).reshape(A_dims, order='F'))
        B_np = np.asfortranarray(np.random.rand(B_nelm).reshape(B_dims, order='F'))

        return A_np, B_np
    
    def test_tensor_subtraction(self):
        A_np, B_np = self.create_rand_np_tensors()

        A = pystarm.Tensor(A_np, A_np.ndim, A_np.shape)
        B = pystarm.Tensor(B_np, B_np.ndim, B_np.shape)

        # Compare results for ALL modes to check for an indexing offset
        C = pystarm.tensor_minus_tensor(A, B)
        C_np = A_np - B_np
        C_pystarm = np.frombuffer(C, dtype=np.float64).reshape(C.getdims(), order='F', copy = False)


        self.assertTrue(np.allclose(C_np, C_pystarm))

    def test_tensor_addition(self):
        '''Test the tensor_plus_tensor function on order 4 tensors.'''
        A_np, B_np = self.create_rand_np_tensors()
        n=5
        C_dims = (n,n,n,n)
        C_nelm = math.prod(B_np.shape)
        C_ndim = len(B_np.shape)
        C_np = np.asfortranarray(np.random.rand(C_nelm).reshape(C_dims, order='F'))

        A = pystarm.Tensor(A_np, A_np.ndim, A_np.shape)
        B = pystarm.Tensor(B_np, B_np.ndim, B_np.shape)
        C = pystarm.Tensor(C_np, C_ndim, C_dims)

        D = pystarm.tensor_plus_tensor(A,B,C)
        D_np = A_np + B_np + C_np
        D_pystarm = np.frombuffer(D, dtype=np.float64).reshape(D.getdims(), order='F', copy = False)

        self.assertTrue(np.allclose(D_np, D_pystarm))

    def test_tensor_addition_two_operands(self):
        '''Test the two-tensor overload of tensor_plus_tensor.'''
        A_np, B_np = self.create_rand_np_tensors()

        A = pystarm.Tensor(A_np, A_np.ndim, A_np.shape)
        B = pystarm.Tensor(B_np, B_np.ndim, B_np.shape)

        C = pystarm.tensor_plus_tensor(A, B)
        C_np = A_np + B_np
        C_pystarm = np.frombuffer(C, dtype=np.float64).reshape(C.getdims(), order='F', copy=False)

        self.assertTrue(np.allclose(C_np, C_pystarm))
        C.clear()


class SlicewiseMatmulDiagTestCase(unittest.TestCase):
    """Tests for slicewise_matmul(Tensor, Matrix, inv_flag, ten_left).

    This overload scales each frontal slice of a tensor by a diagonal matrix held
    compactly as one column of S per slice. Four combinations are reachable:
    multiply/divide crossed with left/right. The division (inv_flag) path was
    previously unreachable from Python and contained an out-of-bounds write: both
    loops that filled the reciprocal matrix ran to S.buflen = nrow*ncol instead of
    to nrow and ncol, and Matrix::set does no bounds checking.

    Tolerances are explicit rather than np.allclose's defaults because these are
    exact scalar products and quotients; anything looser than a few ulp would hide
    an operand mix-up.
    """
    RTOL = 1e-14
    ATOL = 1e-14

    def make(self, m, n, L, seed):
        rng = np.random.default_rng(seed)
        U_np = np.asfortranarray(rng.standard_normal((m, n, L)))
        U = pystarm.Tensor(U_np, U_np.ndim, U_np.shape)
        return rng, U_np, U

    @staticmethod
    def as_np(T):
        return np.frombuffer(T, dtype=np.float64).reshape(T.getdims(), order='F').copy()

    @staticmethod
    def as_mat(a):
        a = np.asfortranarray(a)
        return pystarm.Matrix(a, a.shape[0], a.shape[1])

    def test_right_multiply_by_diagonal(self):
        """U * diag(s) scales column j of each slice by s[j]"""
        rng, U_np, U = self.make(4, 3, 5, 0)
        S_np = rng.standard_normal((3, 5))
        S = self.as_mat(S_np)

        got = self.as_np(pystarm.slicewise_matmul(U, S, inv_flag=False, ten_left=True))
        self.assertTrue(np.allclose(got, U_np * S_np[None, :, :],
                                    rtol=self.RTOL, atol=self.ATOL))

    def test_right_divide_by_diagonal(self):
        """U * diag(1/s) divides column j of each slice by s[j]"""
        rng, U_np, U = self.make(4, 3, 5, 1)
        S_np = rng.standard_normal((3, 5))
        S = self.as_mat(S_np)

        got = self.as_np(pystarm.slicewise_matmul(U, S, inv_flag=True, ten_left=True))
        self.assertTrue(np.allclose(got, U_np / S_np[None, :, :],
                                    rtol=self.RTOL, atol=self.ATOL))

    def test_left_multiply_by_diagonal(self):
        """diag(s) * U scales row i of each slice by s[i]"""
        rng, U_np, U = self.make(4, 3, 5, 2)
        S_np = rng.standard_normal((4, 5))
        S = self.as_mat(S_np)

        got = self.as_np(pystarm.slicewise_matmul(U, S, inv_flag=False, ten_left=False))
        self.assertTrue(np.allclose(got, U_np * S_np[:, None, :],
                                    rtol=self.RTOL, atol=self.ATOL))

    def test_left_divide_by_diagonal(self):
        """diag(1/s) * U divides row i of each slice by s[i]"""
        rng, U_np, U = self.make(4, 3, 5, 3)
        S_np = rng.standard_normal((4, 5))
        S = self.as_mat(S_np)

        got = self.as_np(pystarm.slicewise_matmul(U, S, inv_flag=True, ten_left=False))
        self.assertTrue(np.allclose(got, U_np / S_np[:, None, :],
                                    rtol=self.RTOL, atol=self.ATOL))

    def test_divide_by_zero_singular_value_is_pseudo_inverse(self):
        """A zero entry of S yields zero, not an infinity

        The reciprocal is only defined for nonzero entries. The kernel adopts the
        Moore-Penrose convention 1/s if s != 0 else 0, matching how the reference
        MATLAB (facewiseSVDJacobian.m) guards its own reciprocal. The point of the
        test is that the output stays finite: an unguarded reciprocal would fill
        downstream tensors with Inf and NaN.
        """
        rng, U_np, U = self.make(4, 3, 5, 4)
        S_np = rng.standard_normal((3, 5))
        S_np[1, 2] = 0.0
        S = self.as_mat(S_np)

        got = self.as_np(pystarm.slicewise_matmul(U, S, inv_flag=True, ten_left=True))

        bcast = S_np[None, :, :]
        expected = np.divide(U_np, bcast, out=np.zeros_like(U_np), where=(bcast != 0))
        self.assertTrue(np.isfinite(got).all())
        self.assertTrue(np.allclose(got, expected, rtol=self.RTOL, atol=self.ATOL))

    def test_divide_is_inverse_of_multiply(self):
        """Dividing undoes multiplying when every singular value is nonzero"""
        rng, U_np, U = self.make(6, 4, 3, 5)
        # Keep the entries away from zero so the round trip is well conditioned.
        S_np = rng.uniform(0.5, 2.0, size=(4, 3))
        S = self.as_mat(S_np)

        prod = pystarm.slicewise_matmul(U, S, inv_flag=False, ten_left=True)
        back = pystarm.slicewise_matmul(prod, S, inv_flag=True, ten_left=True)
        self.assertTrue(np.allclose(self.as_np(back), U_np, rtol=1e-13, atol=1e-13))
        prod.clear()
        back.clear()


class SVDGradientTestCase(unittest.TestCase):
    """Finite-difference verification of the star-M SVD gradient helpers.

    The helpers in star-M-opt/svd_gradient_calculations.py are the adjoints of the
    Jacobians of the slicewise SVD factors with respect to the input tensor, i.e.
    MATLAB's JacU.AT / JacS.AT / JacV.AT in tensorSVD/facewiseSVDJacobian.m from
    https://github.com/elizabethnewman/star-M-opt. The property that defines them is

        <d(factor)/dA [x], y>_F  ==  <x, calc_*_gradient(U, S, VT, y)>_F

    for every perturbation direction x of A's shape and every cotangent y of the
    factor's shape. Checking that identity against a central difference of the
    factor is the only test that actually catches a sign or transpose error; an
    eyeball comparison against the MATLAB source does not, as the F-orientation bug
    this suite was written to catch demonstrated (it flipped the sign of both the U
    and the V gradient and nothing else noticed).

    Two separate things are checked, because they can fail independently:

      1. test_matches_matlab_reference_formula: the pystarm implementation agrees
         with a direct NumPy transcription of the MATLAB formulas. This validates
         the port and holds for truncated factors too.
      2. test_*_gradient_matches_finite_differences: the formula itself is the true
         adjoint. This holds only when k == min(m, n).

    KNOWN LIMITATION, pinned by test_truncated_u_and_vt_are_not_the_true_adjoint:
    for a truncated SVD (k < min(m, n)) the U and VT adjoints are NOT the true
    adjoints. The derivative of a truncated singular vector has contributions from
    the discarded singular directions i > k, which the formula's F (restricted to
    the retained k x k block) omits. The reference MATLAB has the same limitation:
    facewiseSVDJacobian.m says "assume matrices are full and full-rank" in its
    header, its own CheckAdjoint tests only exercise the economy case, yet
    objectiveFunctions/objFctnLowRank.m feeds it truncated factors anyway. The
    singular-value adjoint stays exact under truncation.
    """

    # Central differences on an SVD factor converge as O(h^2) until roundoff takes
    # over. At h = 1e-5 the measured relative error across these cases is at worst
    # ~2e-8, so 1e-6 leaves about two decades of headroom while still being tight
    # enough to catch a sign flip (which shows up as a relative error of 2.0).
    FD_STEP = 1e-5
    FD_TOL = 1e-6

    @staticmethod
    def _tensor(a):
        a = np.asfortranarray(a, dtype=np.float64)
        return pystarm.Tensor(a, a.ndim, list(a.shape))

    @staticmethod
    def _as_np(T):
        return np.frombuffer(T, dtype=np.float64).reshape(T.getdims(), order='F').copy()

    @staticmethod
    def _ref_F(s):
        """F[i,j] = 1/(s_j^2 - s_i^2) off the diagonal, 0 where the values coincide."""
        s2 = s ** 2
        D = s2[None, :] - s2[:, None]
        out = np.zeros_like(D)
        np.divide(1.0, D, out=out, where=(D != 0))
        return out

    @classmethod
    def _ref_AT_U(cls, u, s, v, y):
        F, S, m = cls._ref_F(s), np.diag(s), u.shape[0]
        t1 = u @ ((F * (u.T @ y - y.T @ u)) @ S)
        t2 = (np.eye(m) - u @ u.T) @ (y / s[None, :])
        return (t1 + t2) @ v.T

    @classmethod
    def _ref_AT_S(cls, u, s, v, y):
        return u @ (np.eye(len(s)) * y) @ v.T

    @classmethod
    def _ref_AT_V(cls, u, s, v, y):
        F, S, n = cls._ref_F(s), np.diag(s), v.shape[0]
        t1 = (S @ (F * (v.T @ y - y.T @ v))) @ v.T
        t2 = (y / s[None, :]).T @ (np.eye(n) - v @ v.T)
        return u @ (t1 + t2)

    @staticmethod
    def _np_svd_aligned(A_np, k, U_base):
        """Slicewise economy SVD, each slice sign-aligned to a base factorization.

        LAPACK fixes the sign of each singular vector arbitrarily, so the factors at
        A + h*x and A - h*x can come back with opposite signs and wreck the central
        difference. For small h each perturbed vector is close to its unperturbed
        counterpart, so flipping whenever the inner product is negative recovers the
        continuous branch. U and V must be flipped together to keep U S V^T == A.
        """
        m, n, L = A_np.shape
        U = np.empty((m, k, L))
        S = np.empty((k, L))
        Vt = np.empty((k, n, L))
        for l in range(L):
            u, s, vt = np.linalg.svd(A_np[:, :, l], full_matrices=False)
            u, s, vt = u[:, :k].copy(), s[:k].copy(), vt[:k, :].copy()
            for j in range(k):
                if np.dot(u[:, j], U_base[:, j, l]) < 0:
                    u[:, j] = -u[:, j]
                    vt[j, :] = -vt[j, :]
            U[:, :, l] = u
            S[:, l] = s
            Vt[:, :, l] = vt
        return U, S, Vt

    def _setup(self, m, n, L, k, seed):
        """Factor a random tensor and draw a direction x plus one cotangent per factor."""
        rng = np.random.default_rng(seed)
        A_np = np.asfortranarray(rng.standard_normal((m, n, L)))
        A = self._tensor(A_np)
        U, S, VT = pystarm.slicewise_svdx(A, k)
        cotangents = {
            'U': rng.standard_normal((m, k, L)),
            'S': rng.standard_normal((k, k, L)),
            'VT': rng.standard_normal((k, n, L)),
        }
        x = rng.standard_normal((m, n, L))
        return A_np, A, U, S, VT, cotangents, x

    def _analytic(self, U, S, VT, cotangents):
        gU = svdgrad.calc_U_gradient(U, S, VT, self._tensor(cotangents['U']))
        gS = svdgrad.calc_S_gradient(U, S, VT, self._tensor(cotangents['S']))
        gV = svdgrad.calc_VT_gradient(U, S, VT, self._tensor(cotangents['VT']))
        out = {'U': self._as_np(gU), 'S': self._as_np(gS), 'VT': self._as_np(gV)}
        for g in (gU, gS, gV):
            g.clear()
        return out

    def _fd_rel_errors(self, A_np, U_np, k, cotangents, x, analytic, h):
        """Relative gap between <x, analytic> and <d(factor)[x], y> for each factor."""
        L = A_np.shape[2]
        Up, Sp, Vtp = self._np_svd_aligned(A_np + h * x, k, U_np)
        Um, Sm, Vtm = self._np_svd_aligned(A_np - h * x, k, U_np)
        dU = (Up - Um) / (2 * h)
        dS = (Sp - Sm) / (2 * h)
        dVt = (Vtp - Vtm) / (2 * h)

        # The S adjoint only sees the diagonal of its cotangent, which is what pairs
        # with the derivative of the singular values themselves.
        diag_y = np.array([[cotangents['S'][i, i, l] for l in range(L)] for i in range(k)])
        fd = {'U': np.sum(dU * cotangents['U']),
              'S': np.sum(dS * diag_y),
              'VT': np.sum(dVt * cotangents['VT'])}

        errors = {}
        for key, value in fd.items():
            a = float(np.sum(x * analytic[key]))
            errors[key] = abs(a - value) / max(1.0, abs(value))
        return errors

    # ---- shapes exercised: wide, tall and square, all at k = min(m, n) ----
    FULL_RANK_CASES = [(4, 6, 3, 4, 0), (6, 4, 3, 4, 1), (5, 5, 3, 5, 2),
                       (3, 7, 4, 3, 3), (7, 3, 4, 3, 4)]

    def test_u_gradient_matches_finite_differences(self):
        """JacU.AT satisfies the adjoint identity against a central difference"""
        self._check_fd('U')

    def test_s_gradient_matches_finite_differences(self):
        """JacS.AT satisfies the adjoint identity against a central difference"""
        self._check_fd('S')

    def test_vt_gradient_matches_finite_differences(self):
        """JacV.AT satisfies the adjoint identity against a central difference"""
        self._check_fd('VT')

    def _check_fd(self, factor):
        for (m, n, L, k, seed) in self.FULL_RANK_CASES:
            with self.subTest(m=m, n=n, L=L, k=k, seed=seed, factor=factor):
                A_np, A, U, S, VT, cot, x = self._setup(m, n, L, k, seed)
                U_np = self._as_np(U)
                analytic = self._analytic(U, S, VT, cot)
                err = self._fd_rel_errors(A_np, U_np, k, cot, x, analytic, self.FD_STEP)[factor]
                # A is a NumPy-backed Tensor and must never be cleared; U/S/VT are
                # C++ allocations and must be.
                U.clear(); S.clear(); VT.clear()
                self.assertLess(
                    err, self.FD_TOL,
                    f"{factor} gradient disagrees with central differences: "
                    f"relative error {err:.3e} at h={self.FD_STEP:g}. A value near 2.0 "
                    f"means the gradient has the wrong sign.")

    def test_matches_matlab_reference_formula(self):
        """All three adjoints reproduce a NumPy transcription of facewiseSVDJacobian.m

        This is the port-fidelity check and is expected to hold to machine precision
        for truncated factors as well as full-rank ones.
        """
        cases = self.FULL_RANK_CASES + [(6, 8, 2, 3, 5), (8, 6, 2, 3, 6), (6, 6, 2, 2, 7)]
        for (m, n, L, k, seed) in cases:
            with self.subTest(m=m, n=n, L=L, k=k, seed=seed):
                A_np, A, U, S, VT, cot, x = self._setup(m, n, L, k, seed)
                U_np, S_np, Vt_np = self._as_np(U), self._as_np(S), self._as_np(VT)
                analytic = self._analytic(U, S, VT, cot)
                U.clear(); S.clear(); VT.clear()

                expected = {key: np.empty((m, n, L)) for key in analytic}
                for l in range(L):
                    u, s, v = U_np[:, :, l], S_np[:, l], Vt_np[:, :, l].T
                    expected['U'][:, :, l] = self._ref_AT_U(u, s, v, cot['U'][:, :, l])
                    expected['S'][:, :, l] = self._ref_AT_S(u, s, v, cot['S'][:, :, l])
                    expected['VT'][:, :, l] = self._ref_AT_V(u, s, v, cot['VT'][:, :, l].T)

                for key in ('U', 'S', 'VT'):
                    self.assertTrue(
                        np.allclose(analytic[key], expected[key], rtol=1e-12, atol=1e-12),
                        f"{key} gradient does not match the MATLAB reference formula")

    def test_truncated_singular_value_gradient_is_still_exact(self):
        """JacS.AT remains the true adjoint even for a truncated SVD

        Singular values, unlike singular vectors, do not depend on the discarded
        subspace to first order, so truncation does not degrade this one.
        """
        for (m, n, L, k, seed) in [(6, 8, 2, 3, 5), (8, 6, 2, 3, 6), (6, 6, 2, 2, 7)]:
            with self.subTest(m=m, n=n, L=L, k=k, seed=seed):
                A_np, A, U, S, VT, cot, x = self._setup(m, n, L, k, seed)
                U_np = self._as_np(U)
                analytic = self._analytic(U, S, VT, cot)
                err = self._fd_rel_errors(A_np, U_np, k, cot, x, analytic, self.FD_STEP)['S']
                U.clear(); S.clear(); VT.clear()
                self.assertLess(err, self.FD_TOL)

    def test_truncated_u_and_vt_are_not_the_true_adjoint(self):
        """Pin the known truncation limitation of the reference formula

        This test asserts that the U and VT adjoints DISAGREE with finite
        differences when k < min(m, n), and that the disagreement does not shrink
        when the step is refined. The h-independence is what distinguishes a genuine
        formula error from finite-difference noise: a correct adjoint's error would
        fall by roughly 100x between h=1e-4 and h=1e-5.

        If someone extends the formula to handle truncation properly, this test will
        start failing. That is the intent: delete it and update the docstrings in
        star-M-opt/svd_gradient_calculations.py and this class at the same time.
        """
        for (m, n, L, k, seed) in [(6, 8, 2, 3, 5), (8, 6, 2, 3, 6), (6, 6, 2, 2, 7)]:
            A_np, A, U, S, VT, cot, x = self._setup(m, n, L, k, seed)
            U_np = self._as_np(U)
            analytic = self._analytic(U, S, VT, cot)
            coarse = self._fd_rel_errors(A_np, U_np, k, cot, x, analytic, 1e-4)
            fine = self._fd_rel_errors(A_np, U_np, k, cot, x, analytic, 1e-5)
            U.clear(); S.clear(); VT.clear()

            for key in ('U', 'VT'):
                with self.subTest(m=m, n=n, k=k, seed=seed, factor=key):
                    self.assertGreater(
                        fine[key], 1e-3,
                        f"{key} adjoint unexpectedly agrees with finite differences "
                        f"under truncation; the limitation may have been fixed")
                    # Error plateaus instead of falling ~100x: not FD noise.
                    self.assertGreater(fine[key], 0.1 * coarse[key])


# ADD A PYTTB TEST CASE FOR CONTRACTION, LOOK AT TTT FUNCTION AND CONTRACT ALL MODES BUT K.
class MatrixExpTestCase(unittest.TestCase):
    """Tests for the matrix exponential kernels.

    The assertions below pass explicit tolerances rather than inheriting
    np.allclose's defaults (rtol=1e-5, atol=1e-8), which are many decades
    looser than anything these kernels should produce. The tolerances are
    anchored on the measured behaviour: backward-stable Pade lands at
    ~1e-15 relative, and the Schur reuse route at ~1e-13 because of the two
    back-transformations through Q.
    """

    # Relative forward error tolerance for a direct (non-reused) evaluation.
    RTOL_DIRECT = 1e-13
    # Relative forward error tolerance for the Schur reuse route.
    RTOL_REUSE = 1e-11
    # Departure from the orthogonal group; this is the binding requirement.
    ATOL_ORTH = 1e-12

    @staticmethod
    def _tomat(R):
        """Copy a pystarm result buffer into a NumPy array and free it."""
        X = np.frombuffer(R, dtype=np.float64).reshape(R.getdims(), order='F',
                                                       copy=True)
        R.clear()
        return X

    @staticmethod
    def _rand_skew(n, seed):
        rng = np.random.default_rng(seed)
        B = rng.standard_normal((n, n))
        return np.asfortranarray(B - B.T)

    def _relerr(self, X, ref):
        nrm = np.linalg.norm(ref)
        if nrm == 0.0:
            return np.linalg.norm(X)
        return np.linalg.norm(X - ref) / nrm

    def test_expm_general_matches_scipy(self):
        """Pade path agrees with scipy.linalg.expm on general input"""
        for n in (1, 2, 5, 17, 32):
            rng = np.random.default_rng(100 + n)
            Apy = np.asfortranarray(rng.standard_normal((n, n)))
            Amat = pystarm.Matrix(Apy, n, n)
            X = self._tomat(pystarm.expm(Amat))
            err = self._relerr(X, scipy.linalg.expm(Apy))
            self.assertLess(err, self.RTOL_DIRECT)

    def test_expm_general_scaling_and_squaring(self):
        """Pade path is accurate for a large-norm matrix (several squarings)"""
        n = 24
        rng = np.random.default_rng(7)
        Apy = np.asfortranarray(20.0 * rng.standard_normal((n, n)))
        Amat = pystarm.Matrix(Apy, n, n)
        X = self._tomat(pystarm.expm(Amat))
        err = self._relerr(X, scipy.linalg.expm(Apy))
        self.assertLess(err, self.RTOL_DIRECT)

    def test_expm_scalar_multiplier(self):
        """exp(t*A) matches expm of the explicitly scaled matrix"""
        n = 12
        rng = np.random.default_rng(11)
        Apy = np.asfortranarray(rng.standard_normal((n, n)))
        Amat = pystarm.Matrix(Apy, n, n)
        for t in (-1.7, 0.0, 0.25, 3.0):
            X = self._tomat(pystarm.expm(Amat, t))
            err = self._relerr(X, scipy.linalg.expm(t * Apy))
            self.assertLess(err, self.RTOL_DIRECT)

    def test_expm_skew_matches_scipy(self):
        """Both paths agree with scipy.linalg.expm on skew input, odd and even n"""
        for n in (1, 2, 3, 8, 9, 31, 64):
            Spy = self._rand_skew(n, 200 + n)
            Smat = pystarm.Matrix(Spy, n, n)
            ref = scipy.linalg.expm(Spy)

            Xp = self._tomat(pystarm.expm(Smat))
            self.assertLess(self._relerr(Xp, ref), self.RTOL_DIRECT)

            Xs = self._tomat(pystarm.expm_skew(Smat))
            self.assertLess(self._relerr(Xs, ref), self.RTOL_REUSE)

    def test_expm_skew_orthogonality(self):
        """Both paths land on the orthogonal group for skew input"""
        for n in (2, 9, 64):
            Spy = self._rand_skew(n, 300 + n)
            Smat = pystarm.Matrix(Spy, n, n)
            for X in (self._tomat(pystarm.expm(Smat)),
                      self._tomat(pystarm.expm_skew(Smat)),
                      self._tomat(pystarm.cayley(Smat))):
                dev = np.linalg.norm(X.T @ X - np.eye(n))
                self.assertLess(dev, self.ATOL_ORTH)

    def test_skew_schur_decomposition(self):
        """skew_schur returns an orthogonal Q and angles reproducing S"""
        for n in (7, 16):
            Spy = self._rand_skew(n, 400 + n)
            Smat = pystarm.Matrix(Spy, n, n)
            Q, theta = pystarm.skew_schur(Smat)
            Qpy = np.frombuffer(Q, dtype=np.float64).reshape(Q.getdims(),
                                                             order='F', copy=True)
            self.assertLess(np.linalg.norm(Qpy.T @ Qpy - np.eye(n)),
                            self.ATOL_ORTH)

            # Rebuild S = Q T Q^T from the reported block angles.
            T = np.zeros((n, n), order='F')
            i = 0
            while i < n:
                if i + 1 < n and theta[i] != 0.0:
                    T[i, i + 1] = theta[i]
                    T[i + 1, i] = -theta[i]
                    i += 2
                else:
                    i += 1
            self.assertLess(self._relerr(Qpy @ T @ Qpy.T, Spy), self.RTOL_REUSE)

            # The input must not have been destroyed.
            Spy2 = np.frombuffer(Smat, dtype=np.float64).reshape(Smat.getdims(),
                                                                 order='F', copy=False)
            self.assertEqual(np.allclose(Spy, Spy2, rtol=0, atol=0), True)
            Q.clear()

    def test_expm_reuse_across_halved_steps(self):
        """One decomposition reproduces fresh evaluation at every halved step"""
        n = 48
        Spy = self._rand_skew(n, 500)
        Smat = pystarm.Matrix(Spy, n, n)
        Q, theta = pystarm.skew_schur(Smat)
        for j in range(10):
            t = 2.0 ** (-j)
            X = self._tomat(pystarm.expm_skew_apply(Q, theta, t))
            ref = scipy.linalg.expm(t * Spy)
            self.assertLess(self._relerr(X, ref), self.RTOL_REUSE)
            self.assertLess(np.linalg.norm(X.T @ X - np.eye(n)), self.ATOL_ORTH)
        Q.clear()

    def test_expm_multi_reuse_matches_direct(self):
        """expm_multi gives the same answers with and without the reuse path"""
        n = 33
        Spy = self._rand_skew(n, 600)
        Smat = pystarm.Matrix(Spy, n, n)
        ts = [2.0 ** (-j) for j in range(12)]

        reused = pystarm.expm_multi(Smat, ts, pystarm.RETRACT_EXPM, 1)
        direct = pystarm.expm_multi(Smat, ts, pystarm.RETRACT_EXPM, len(ts) + 1)
        for t, Ra, Rb in zip(ts, reused, direct):
            Xa = self._tomat(Ra)
            Xb = self._tomat(Rb)
            ref = scipy.linalg.expm(t * Spy)
            self.assertLess(self._relerr(Xa, ref), self.RTOL_REUSE)
            self.assertLess(self._relerr(Xb, ref), self.RTOL_DIRECT)

    def test_cayley_matches_reference(self):
        """Cayley matches a direct solve, and reuses the decomposition"""
        n = 20
        Spy = self._rand_skew(n, 700)
        Smat = pystarm.Matrix(Spy, n, n)
        eye = np.eye(n)
        ts = [1.0, 0.5, 0.25, 0.125]

        for t in ts:
            X = self._tomat(pystarm.cayley(Smat, t))
            ref = np.linalg.solve(eye - 0.5 * t * Spy, eye + 0.5 * t * Spy)
            self.assertLess(self._relerr(X, ref), self.RTOL_DIRECT)
            self.assertLess(np.linalg.norm(X.T @ X - eye), self.ATOL_ORTH)

        Q, theta = pystarm.skew_schur(Smat)
        for t in ts:
            X = self._tomat(pystarm.cayley_skew_apply(Q, theta, t))
            ref = np.linalg.solve(eye - 0.5 * t * Spy, eye + 0.5 * t * Spy)
            self.assertLess(self._relerr(X, ref), self.RTOL_REUSE)
        Q.clear()

    def test_is_skew_symmetric(self):
        """The dispatch test accepts skew matrices and rejects general ones"""
        n = 9
        Spy = self._rand_skew(n, 800)
        rng = np.random.default_rng(801)
        Apy = np.asfortranarray(rng.standard_normal((n, n)))
        Zpy = np.asfortranarray(np.zeros((n, n)))
        self.assertEqual(pystarm.is_skew_symmetric(pystarm.Matrix(Spy, n, n)), True)
        self.assertEqual(pystarm.is_skew_symmetric(pystarm.Matrix(Apy, n, n)), False)
        self.assertEqual(pystarm.is_skew_symmetric(pystarm.Matrix(Zpy, n, n)), True)

    def test_expm_edge_cases(self):
        """n = 1 and n = 2, and the zero matrix"""
        one = np.asfortranarray(np.array([[0.7]]))
        X = self._tomat(pystarm.expm(pystarm.Matrix(one, 1, 1)))
        self.assertLess(abs(X[0, 0] - np.exp(0.7)), 1e-15 * np.exp(0.7))

        zero1 = np.asfortranarray(np.zeros((1, 1)))
        X = self._tomat(pystarm.expm(pystarm.Matrix(zero1, 1, 1)))
        self.assertEqual(X[0, 0], 1.0)

        rot = np.asfortranarray(np.array([[0.0, 1.3], [-1.3, 0.0]]))
        ref = np.array([[np.cos(1.3), np.sin(1.3)],
                        [-np.sin(1.3), np.cos(1.3)]])
        for f in (pystarm.expm, pystarm.expm_skew):
            X = self._tomat(f(pystarm.Matrix(rot, 2, 2)))
            self.assertLess(self._relerr(X, ref), self.RTOL_DIRECT)

        zero2 = np.asfortranarray(np.zeros((2, 2)))
        X = self._tomat(pystarm.expm(pystarm.Matrix(zero2, 2, 2)))
        self.assertLess(self._relerr(X, np.eye(2)), self.RTOL_DIRECT)

    def test_expm_rejects_non_square(self):
        """A non-square input raises rather than returning silently"""
        Apy = np.asfortranarray(np.random.rand(6).reshape((3, 2), order='F'))
        Amat = pystarm.Matrix(Apy, 3, 2)
        with self.assertRaises(RuntimeError):
            pystarm.expm(Amat)

if __name__ == "__main__":
    unittest.main(verbosity=2)
