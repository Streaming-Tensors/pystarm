import unittest
import numpy as np
import pyttb as ttb
import os
import itertools
import gc
import time
import math
from alg import starM_product, starM_product_ttb, tensor_contraction_product_ttb, tensor_contract_multiply
import pystarm


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

    def test_third_order_product_transpose(self):
       '''Test a third order starM product while transposing both slices'''
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

       C_w_transpose = starM_product(B,A,M, A_transpose=True, B_transpose=True)
       C_wo_transpose = starM_product(A,B,M,A_transpose=False, B_transpose=False)

       self.assertFalse(np.allclose(C_w_transpose, C_wo_transpose.T))

class ContractionTestCase(unittest.TestCase):
    def test_order_three_contraction(self):
        '''
        Test a tensor contraction with all modes of two mode-3 tensors.
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
            C_ttb = tensor_contraction_product_ttb(A, B, mode)
            flags.append(np.allclose(C_pystarm, C_ttb))
    def test_order_four_contractions(self):
        '''
        Test a tensor contractions with all modes of two mode-4 tensors.
        '''
        modes = [0, 1, 2, 3]
        flags = []
        for mode in modes:
            n_k = np.random.randint(2, 10)
            p = np.random.randint(2,10)
            m = np.random.randint(2,10)

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
            C_ttb = tensor_contraction_product_ttb(A, B, mode)
            flags.append(np.allclose(C_pystarm, C_ttb))

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
      
       

# ADD A PYTTB TEST CASE FOR CONTRACTION, LOOK AT TTT FUNCTION AND CONTRACT ALL MODES BUT K.

if __name__ == "__main__":
    unittest.main(verbosity=2)
