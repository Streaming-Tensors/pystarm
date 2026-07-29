import unittest
import itertools
import numpy as np
import pystarm


def _to_tensor(arr):
    arr = np.asfortranarray(arr, dtype=np.float64)
    return pystarm.Tensor(arr, arr.ndim, arr.shape), arr


def _to_matrix(arr):
    arr = np.asfortranarray(arr, dtype=np.float64)
    return pystarm.Matrix(arr, arr.shape[0], arr.shape[1]), arr


def _from_tensor(ten):
    return np.frombuffer(ten, dtype=np.float64).reshape(
        ten.getdims(), order='F', copy=False
    )


def _np_slicewise_matmul(A_np, B_np, transpose_A=False, transpose_B=False):
    """Reference: slicewise matmul over trailing modes, collapsed to 3D."""
    A3 = A_np.reshape(A_np.shape[0], A_np.shape[1], -1, order='F')
    B3 = B_np.reshape(B_np.shape[0], B_np.shape[1], -1, order='F')
    nslices = A3.shape[2]
    slices = []
    for i in range(nslices):
        a = A3[:, :, i].T if transpose_A else A3[:, :, i]
        b = B3[:, :, i].T if transpose_B else B3[:, :, i]
        slices.append(a @ b)
    return np.asfortranarray(np.stack(slices, axis=-1))


def _np_slicewise_matmul_diag(U_np, S_np):
    """Reference: right-multiply each slice of U by diag(S[:, i])."""
    U3 = U_np.reshape(U_np.shape[0], U_np.shape[1], -1, order='F')
    nslices = U3.shape[2]
    slices = [U3[:, :, i] @ np.diag(S_np[:, i]) for i in range(nslices)]
    return np.asfortranarray(np.stack(slices, axis=-1))


class SlicewiseMatmulTensorTensor(unittest.TestCase):
    """Tests for slicewise_matmul(Tensor, Tensor, transpose_A, transpose_B)."""

    def _check(self, A_shape, B_shape, transpose_A, transpose_B):
        rng = np.random.default_rng(0)
        A_np = rng.standard_normal(A_shape)
        B_np = rng.standard_normal(B_shape)

        A_ten, A_np = _to_tensor(A_np)
        B_ten, B_np = _to_tensor(B_np)

        C_ten = pystarm.slicewise_matmul(A_ten, B_ten, transpose_A, transpose_B)
        C_cpp = _from_tensor(C_ten)
        C_ref = _np_slicewise_matmul(A_np, B_np, transpose_A, transpose_B)

        # slicewise_matmul collapses trailing modes into one; compare as 3D.
        C_cpp3 = C_cpp.reshape(C_cpp.shape[0], C_cpp.shape[1], -1, order='F')
        self.assertEqual(C_cpp3.shape, C_ref.shape)
        self.assertTrue(np.allclose(C_cpp3, C_ref))

    def test_3d_no_transpose(self):
        self._check((5, 4, 3), (4, 6, 3), False, False)

    def test_3d_transpose_A(self):
        # A slices are (4, 5); op(A) is (5, 4). B is (5, 6, 3). Inner = 5.
        self._check((4, 5, 3), (5, 6, 3), True, False)

    def test_3d_transpose_B(self):
        # A slices are (5, 4); B slices are (6, 4); op(B) is (4, 6). Inner = 4.
        self._check((5, 4, 3), (6, 4, 3), False, True)

    def test_3d_transpose_both(self):
        # op(A): (5, 4), op(B): (4, 6). Stored A: (4, 5), stored B: (6, 4).
        self._check((4, 5, 3), (6, 4, 3), True, True)

    def test_4d_no_transpose(self):
        self._check((5, 4, 3, 2), (4, 6, 3, 2), False, False)

    def test_4d_transpose_both(self):
        self._check((4, 5, 3, 2), (6, 4, 3, 2), True, True)

    def test_square_slices(self):
        self._check((4, 4, 5), (4, 4, 5), False, False)

    def test_single_slice(self):
        self._check((5, 4, 1), (4, 6, 1), False, False)

    def test_known_values(self):
        # Two 2x2 slices with hand-checkable arithmetic.
        A = np.zeros((2, 2, 2), order='F')
        A[:, :, 0] = [[1.0, 2.0], [3.0, 4.0]]
        A[:, :, 1] = [[5.0, 6.0], [7.0, 8.0]]
        B = np.zeros((2, 2, 2), order='F')
        B[:, :, 0] = [[1.0, 0.0], [0.0, 1.0]]   # identity
        B[:, :, 1] = [[0.0, 1.0], [1.0, 0.0]]   # swap columns

        A_ten, _ = _to_tensor(A)
        B_ten, _ = _to_tensor(B)

        C_ten = pystarm.slicewise_matmul(A_ten, B_ten, False, False)
        C = _from_tensor(C_ten).reshape(2, 2, 2, order='F')

        np.testing.assert_allclose(C[:, :, 0], A[:, :, 0])
        np.testing.assert_allclose(C[:, :, 1], [[2.0, 1.0], [4.0, 3.0]])


class SlicewiseMatmulTensorDiag(unittest.TestCase):
    """Tests for slicewise_matmul(Tensor, Matrix) with Matrix as diagonal storage.

    S has shape (r, nslices); slice i uses diag(S[:, i]).
    Requires U.dims[1] == S.nrow and U.nslices == S.ncol.
    """

    def _check(self, U_shape):
        rng = np.random.default_rng(1)
        m, r = U_shape[0], U_shape[1]
        nslices = int(np.prod(U_shape[2:]))

        U_np = rng.standard_normal(U_shape)
        S_np = rng.standard_normal((r, nslices))

        U_ten, U_np = _to_tensor(U_np)
        S_mat, S_np = _to_matrix(S_np)

        US_ten = pystarm.slicewise_matmul(U_ten, S_mat)
        US_cpp = _from_tensor(US_ten)
        US_ref = _np_slicewise_matmul_diag(U_np, S_np)

        US_cpp3 = US_cpp.reshape(m, r, -1, order='F')
        self.assertEqual(US_cpp3.shape, US_ref.shape)
        self.assertTrue(np.allclose(US_cpp3, US_ref))

    def test_3d(self):
        self._check((5, 4, 3))

    def test_4d(self):
        self._check((5, 4, 3, 2))

    def test_square(self):
        self._check((4, 4, 5))

    def test_single_slice(self):
        self._check((5, 4, 1))

    def test_tall_slices(self):
        self._check((8, 3, 4))

    def test_wide_slices(self):
        self._check((3, 8, 4))

    def test_known_values(self):
        # Two 2x3 slices scaled by two known diagonals.
        U = np.zeros((2, 3, 2), order='F')
        U[:, :, 0] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        U[:, :, 1] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        S = np.array([[10.0, 100.0],
                      [20.0, 200.0],
                      [30.0, 300.0]], order='F')

        U_ten, _ = _to_tensor(U)
        S_mat, _ = _to_matrix(S)

        US_ten = pystarm.slicewise_matmul(U_ten, S_mat)
        US = _from_tensor(US_ten).reshape(2, 3, 2, order='F')

        expected0 = U[:, :, 0] * S[:, 0][np.newaxis, :]
        expected1 = U[:, :, 1] * S[:, 1][np.newaxis, :]
        np.testing.assert_allclose(US[:, :, 0], expected0)
        np.testing.assert_allclose(US[:, :, 1], expected1)

    def test_zero_diagonal(self):
        # Scaling by an all-zero diagonal should produce zeros.
        U_np = np.random.default_rng(2).standard_normal((4, 3, 5))
        S_np = np.zeros((3, 5))
        U_ten, _ = _to_tensor(U_np)
        S_mat, _ = _to_matrix(S_np)
        US = _from_tensor(pystarm.slicewise_matmul(U_ten, S_mat))
        self.assertTrue(np.allclose(US, 0.0))

    def test_identity_diagonal(self):
        # Scaling by all-ones diagonal returns U unchanged (per slice).
        U_np = np.random.default_rng(3).standard_normal((4, 3, 5))
        S_np = np.ones((3, 5))
        U_ten, U_np = _to_tensor(U_np)
        S_mat, _ = _to_matrix(S_np)
        US = _from_tensor(pystarm.slicewise_matmul(U_ten, S_mat))
        US3 = US.reshape(4, 3, 5, order='F')
        self.assertTrue(np.allclose(US3, U_np))


if __name__ == '__main__':
    unittest.main()
