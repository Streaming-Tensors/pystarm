import time
import numpy as np
import scipy as sp
from scipy.fft import dct
import numpy as np
import matplotlib.pyplot as plt
import pystarm
import pyttb as ttb

def tsvdm_I_compress(A, Ms, ttm_modes, k, verbose=False):
    '''
    Assumes A is a pystarm tensor and Ms is a python list of pystarm matrices
    '''
    print("[tsvdm_I_compress]", "ttm_modes", ttm_modes)
    A_hat = None
    for i in range(len(ttm_modes)):
        # print("[", i, "]")
        mode = ttm_modes[i]

        if verbose:
            n = A.getdims()[mode]
            print("[tsvdm_I_compress]", "Dimension in mode", mode, "is", n)
        
        t0 = time.perf_counter()
        if i == 0:
            A_hat_temp = pystarm.ttm(A, Ms[i], mode) 
        else:
            A_hat_temp = pystarm.ttm(A_hat, Ms[i], mode) 
        t1 = time.perf_counter()

        # y = np.frombuffer(A_hat_temp, dtype=np.float64).reshape(A_hat_temp.getdims(), order='F', copy = False)
        # print(np.linalg.norm(y))

        if A_hat is not None:
            # To prevent calling clear in the first iteration
            # Memory would be cleared in the subsequent iterations, as new memory is allocated with new TTM
            A_hat.clear()
        A_hat = A_hat_temp
        
        if verbose:
            print("[tsvdm_I_compress]", "Time for TTM on mode", mode, ":", t1-t0)

    # if A_hat is None:
        # A_hat = A


    t0 = time.perf_counter()
    if A_hat is None:
        U_hat, S_hat, V_hat = pystarm.slicewise_svdx(A, k)
    else:
        U_hat, S_hat, V_hat = pystarm.slicewise_svdx(A_hat, k)
    # U_hat, S_hat, V_hat = pystarm.slicewise_svd(A_hat)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_I_compress]", "Time for slicewise SVD:", t1-t0)

    # if A_hat is not A:
        # A_hat.clear()
    if A_hat is not None:
        A_hat.clear()

    return (U_hat,S_hat,V_hat)

def tsvdm_II_compress(A, Ms, ttm_modes, pct, verbose=False):
    print("[tsvdm_II_compress]", "ttm_modes", ttm_modes)
    A_hat = None
    for i in range(len(ttm_modes)):
        # print("[", i, "]")
        mode = ttm_modes[i]

        if verbose:
            n = A.getdims()[mode]
            print("[tsvdm_II_compress]", "Dimension in mode", mode, "is", n)
        
        t0 = time.perf_counter()
        if i == 0:
            A_hat_temp = pystarm.ttm(A, Ms[i], mode) 
        else:
            A_hat_temp = pystarm.ttm(A_hat, Ms[i], mode) 
        t1 = time.perf_counter()

        # y = np.frombuffer(A_hat_temp, dtype=np.float64).reshape(A_hat_temp.getdims(), order='F', copy = False)
        # print(np.linalg.norm(y))

        if A_hat is not None:
            # To prevent calling clear in the first iteration
            # Memory would be cleared in the subsequent iterations, as new memory is allocated with new TTM
            A_hat.clear()
        A_hat = A_hat_temp
        
        if verbose:
            print("[tsvdm_II_compress]", "Time for TTM on mode", mode, ":", t1-t0)

    if A_hat is None:
        A_hat = A

    t0 = time.perf_counter()
    Sv = pystarm.slicewise_svdvals(A_hat)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_II_compress]", "Time for slicewise_svdvals:", t1-t0)

    t0 = time.perf_counter()
    ks = pystarm.thresholds(Sv, pct)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_II_compress]", "Time for thresholds:", t1-t0)

    t0 = time.perf_counter()
    U_hat, S_hat, V_hat = pystarm.slicewise_svdks(A_hat, ks)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_II_compress]", "Time for slicewise_svdks:", t1-t0)
    
    Sv.clear()
    if A_hat is not A:
        A_hat.clear()

    return (U_hat,S_hat,V_hat)

def tsvdm_I_reconstruct(U_hat, S_hat, VT_hat, Minvs, ttm_modes, orig_shape, verbose=False):
    t0 = time.perf_counter()
    A_hat = pystarm.slicewise_matmul(U_hat, S_hat, VT_hat)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_I_reconstruct] Time to reconstruct A_hat:", t1-t0)

    # slicewise_matmul always returns a 3D tensor, collapsing all modes >= 2
    # into a single nslices dimension. Reshape back to the original nD shape
    # before applying inverse TTMs (or returning for the eye/empty case).
    if len(A_hat.getdims()) == 3 and len(orig_shape) > 3:
        arr_nd = np.frombuffer(A_hat, dtype=np.float64).reshape(orig_shape, order='F')
        A_hat  = pystarm.Tensor(arr_nd, len(orig_shape), orig_shape)

    if verbose:
        print("[tsvdm_I_reconstruct]", "ttm_modes", ttm_modes)

    if len(ttm_modes) == 0:
        return A_hat

    A_tilde = None
    for i in range(len(ttm_modes)):
        mode = ttm_modes[i]

        if verbose:
            n = A_hat.getdims()[mode]
            print("[tsvdm_I_reconstruct]", "Dimension in mode", mode, "is", n)
        
        t0 = time.perf_counter()
        if i == 0:
            A_tilde_temp = pystarm.ttm(A_hat, Minvs[i], mode) 
        else:
            A_tilde_temp = pystarm.ttm(A_tilde, Minvs[i], mode) 
        t1 = time.perf_counter()

        if A_tilde is not None:
            # To prevent calling clear in the first iteration
            # Memory would be cleared in the subsequent iterations, as new memory is allocated with new TTM
            A_tilde.clear()
        A_tilde = A_tilde_temp
        
        if verbose:
            print("[tsvdm_I_reconstruct]", "Time for TTM on mode", mode, ":", t1-t0)

    A_hat.clear()
    
    return A_tilde

def tsvdm_II_reconstruct(U_hat, S_hat, VT_hat, Minvs, ttm_modes, orig_shape, verbose=False):
    t0 = time.perf_counter()
    A_hat = pystarm.slicewise_matmulks(U_hat, S_hat, VT_hat)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_II_reconstruct] Time to reconstruct A_hat:", t1-t0)

    # slicewise_matmulks always returns a 3D tensor, collapsing all modes >= 2
    # into a single nslices dimension. Reshape back to the original nD shape
    # before applying inverse TTMs (or returning for the eye/empty case).
    if len(A_hat.getdims()) == 3 and len(orig_shape) > 3:
        arr_nd = np.frombuffer(A_hat, dtype=np.float64).reshape(orig_shape, order='F')
        A_hat  = pystarm.Tensor(arr_nd, len(orig_shape), orig_shape)

    if verbose:
        print("[tsvdm_II_reconstruct]", "ttm_modes", ttm_modes)

    if len(ttm_modes) == 0:
        return A_hat

    A_tilde = None
    for i in range(len(ttm_modes)):
        mode = ttm_modes[i]

        if verbose:
            n = A_hat.getdims()[mode]
            print("[tsvdm_II_reconstruct]", "Dimension in mode", mode, "is", n)
        
        t0 = time.perf_counter()
        if i == 0:
            A_tilde_temp = pystarm.ttm(A_hat, Minvs[i], mode) 
        else:
            A_tilde_temp = pystarm.ttm(A_tilde, Minvs[i], mode) 
        t1 = time.perf_counter()

        if A_tilde is not None:
            # To prevent calling clear in the first iteration
            # Memory would be cleared in the subsequent iterations, as new memory is allocated with new TTM
            A_tilde.clear()
        A_tilde = A_tilde_temp
        
        if verbose:
            print("[tsvdm_II_reconstruct]", "Time for TTM on mode", mode, ":", t1-t0)

    A_hat.clear()
    
    return A_tilde

def starM_product(A_np, B_np, M_np, A_transpose=False, B_transpose=False):
    '''
    This function computes the starM Product of any given order >= 3 input tensor and an
    orthogonal transformation matrix M.

    Inputs:
    - A: numpy tensor (ndarray) with ndim >= 3
    - B: numpy tensors (ndarray) with ndim >= 3 
    - M: must be a python array of numpy matrices.

    Returns:
    - C: starM product of A and B
    '''
    # Go ahead and calculate M^(-1), will be used in the last step.
    # Since all matrices are orthogonal, no need to make a list of inverse matrices. just pass in the transpose.
    A_ndim = A_np.ndim
    B_ndim = B_np.ndim

    # Make it such that for order 3, user does not have to pass in a list of numpy arrays for M.
    if(type(M_np) != list and A_ndim == 3):
        M_np = [M_np]

    assert A_ndim >= 3, "A must be a tensor with ndim >= 3"
    assert B_ndim >= 3, "B must be a tensor with ndim >= 3"
    
    A_shape = A_np.shape
    B_shape = B_np.shape

    # Ensure dimensions are correct.
    assert A_shape[A_ndim - 1] == B_shape[B_ndim - 1], "A and B must have the same number of slices."
    assert A_ndim == len(M_np) + 2, "You must have (order - 2) transformation matrices inside of a python list. I.e. order 4 must have a list with 2 transformation matrices."
    assert np.allclose(np.dot(M_np[0], M_np[0].T), np.identity(len(M_np[0]))), "M must be an orthogonal matrix."

    # Convert from numpy to starM Tensor/Matrix for pystarm API.
    A = pystarm.Tensor(A_np, A_ndim, A_shape) # args are array (numpy), ndim/order, and dimensions
    B = pystarm.Tensor(B_np, B_ndim, B_shape)

    # Do initial mode 3 calculation, prevents A_hat and B_hat from going out of range
    M_shape = M_np[0].shape
    M = pystarm.Matrix(M_np[0], M_shape[0], M_shape[1])

    # Move A and B into transform domain
    current_mode = 2 # start with mode 3-1=2
    A_hat = pystarm.ttm(A, M, current_mode) # Args: tensor, matrix, mode (index 0, so if you put 1 it will be mode 1+1=2)
    B_hat = pystarm.ttm(B, M, current_mode)
    # For higher orders (>= 4), move those modes into transform domain. 
    if A_ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            # Check for M's orthogonality.
            assert np.allclose(np.dot(matrix, matrix.T), np.identity(len(matrix))), "All M's must be orthogonal matrices."
            M_shape = matrix.shape
            M = pystarm.Matrix(matrix, M_shape[0], M_shape[1])
            # Move A and B into transform domain
            A_hat = pystarm.ttm(A_hat, M, current_mode) # Args: tensor, matrix, mode (index 0, so if you put 1 it will be mode 1+1=2)
            B_hat = pystarm.ttm(B_hat, M, current_mode)

    # Slicewise multiply A and B
    C_hat = pystarm.slicewise_matmul(A_hat, B_hat, A_transpose, B_transpose)

    current_mode = 2 # reset mode counter
    # Return product to original domain. NOTE: Since all M's are orthogonal, the inverse = transpose. That's what I'm doing here.
    M_inv = pystarm.Matrix(M_np[0].T, M_np[0].shape[1], M_np[0].shape[0])
    C = pystarm.ttm(C_hat, M_inv, current_mode)

    if A_ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            M_inv = pystarm.Matrix(matrix.T, matrix.shape[1], matrix.shape[0])
            C = pystarm.ttm(C, M_inv, current_mode)
    
    # Return to numpy
    C_np = np.frombuffer(C, dtype=np.float64).reshape(C.getdims(), order='F', copy = False)
    
    return C_np

def facewise_product_pyttb(A, B):
    """
    Compute the facewise (frontal-slice-wise) matrix product of two pyttb tensors.
    Inputs:
    - A: pyttb.tensor
    - B: pyttb.tensor
    
    Returns:
    - C: pyttb.tensor
    """
    # Extract underlying numpy arrays
    A_data = A.data 
    B_data = B.data  

    ndim = A_data.ndim

    # Move frontal slice indicies to the front for batched matmul
    # Build the permutation: [2, 3, ..., ndim-1, 0, 1]
    batch_axes = list(range(2, ndim))
    perm_forward = batch_axes + [0, 1]
    
    A_t = np.transpose(A_data, perm_forward)
    B_t = np.transpose(B_data, perm_forward)

    # Batched matrix multiply over all leading (batch) dimensions
    C_t = np.matmul(A_t, B_t)

    # Invert the permutation: move last two axes back to positions 0, 1
    # From (K1, K2, ..., Kn, I, L) -> (I, L, K1, K2, ..., Kn)
    perm_back = [ndim - 2, ndim - 1] + list(range(ndim - 2))
    
    C_data = np.transpose(C_t, perm_back)

    return ttb.tensor(C_data)

def starM_product_ttb(A_np, B_np, M_np):
    '''
    Compute the star-M product using the pyttb library. Used for the test.py script to verify the results of `starM_product`.
    Inputs: 
    - A_np, B_np - numpy tensors (ndarray) with ndim >= 3. 
    - M_np, if ndim == 3, it can be a single numpy matrix. If ndim > 3, must be a python list with ndim - 2 numpy matrices.

    Returns:
    - C = starM product of A and B, numpy tensor (ndarray).
    '''
    # Move to transform domain + calculate facewise product
    ndim = A_np.ndim
    if(type(M_np) != list and ndim == 3):
        M_np = [M_np]
    
    # Move mode 3 into transform domain. 
    current_mode = 2 # Since python does index 0, so mode 3 = 3-1=2
    A_hat = ttb.tensor(A_np, copy=True).ttm(M_np[0], current_mode)
    B_hat = ttb.tensor(B_np, copy=True).ttm(M_np[0], current_mode)

    # Move modes (4, ndim) into transform domain.
    if ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            # Check for M's orthogonality.
            assert np.allclose(np.dot(matrix, matrix.T), np.identity(len(matrix))), "All M's must be orthogonal matrices."

            # Move A and B into transform domain for current_mode
            A_hat = A_hat.ttm(matrix, current_mode)
            B_hat = B_hat.ttm(matrix, current_mode)

    C_hat = facewise_product_pyttb(A_hat, B_hat)

    # Move back to original domain and convert to numpy array
    current_mode = 2
    C = C_hat.ttm(M_np[0].T, current_mode)
    if ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            C = C.ttm(matrix, current_mode)

    return C.data

def tensor_contract_multiply(A_np, B_np, k, naive: bool=False):
    '''
    Takes two numpy tensors, unfolds them along mode k, and computes A_(k) B_(k)^T.
    Returns: 
    Numpy matrix of size n_k X p, where n_k is the size of the kth dimension of A, and p is the kth dimension of B.

    Inputs:
    A_np: Numpy tensor of size n_1 x ... x n_k x ... x n_d
    B_np: Numpy tensor of size n_1 x ... x p x ... x n_d
    k: Mode to unfold tensors on.
    naive: Boolean value which determines whether or not to use the naive tensor contraction algo. False will save time but add overhead, True will take more time with less overhead.
    '''
    A_ndim = A_np.ndim
    B_ndim = B_np.ndim

    assert A_ndim >= 3, "A must be a tensor with ndim >= 3"
    assert B_ndim >= 3, "B must be a tensor with ndim >= 3"
    
    A_shape = A_np.shape
    B_shape = B_np.shape

    A = pystarm.Tensor(A_np, A_ndim, A_shape)
    B = pystarm.Tensor(B_np, B_ndim, B_shape)
    
    # Slicewise multiply A and B
    C = pystarm.tensor_contract_all_but_one(A, B, k, naive)
    # Return to numpy
    C_np = np.frombuffer(C, dtype=np.float64).reshape(C.getdims(), order='F', copy = False)
    
    return C_np


def tensor_contraction_product_ttb(A, B, mode: int) -> np.ndarray:
    """
    Compute the mode-k tensor contraction product: C = A_(k) @ B_(k)^T

    Given two tensors A and B of the same shape, this function:
      1. Unfolds (matricizes) both tensors along the specified mode.
      2. Computes the matrix product A_(k) @ B_(k)^T.

    Parameters
    ----------
    A : ttb.tensor
        First input tensor.
    B : ttb.tensor
        Second input tensor.
    mode : int
        The mode along which to unfold (0-indexed).

    Returns
    -------
    C : np.ndarray
        The resulting matrix of shape (I_k, I_k), where I_k = A.shape[mode].

    Raises
    ------
    ValueError
        If A and B do not have the same shape.
    ValueError
        If mode is out of range for the tensor dimensions.
    """
    # --- Input validation ---
    # if A.shape != B.shape:
    #     raise ValueError(
    #         f"Tensors must have the same shape. Got A.shape={A.shape}, B.shape={B.shape}"
    #     )

    ndims = A.ndim
    if mode < 0 or mode >= ndims:
        raise ValueError(
            f"Mode {mode} is out of range for tensors with {ndims} dimensions "
            f"(valid range: 0 to {ndims - 1})."
        )
    A_ttb = ttb.tensor(A, shape=A.shape)
    B_ttb = ttb.tensor(B, shape=B.shape)

    # --- Mode-k unfolding using pyttb's tenmat ---
    # tenmat(tensor, rdims) unfolds the tensor so that the specified
    # mode(s) become the row dimensions and all remaining modes become columns.
    A_unfolded = A_ttb.to_tenmat(rdims=np.array([mode]))
    del A_ttb
    B_unfolded = B_ttb.to_tenmat(rdims=np.array([mode]))
    del B_ttb

    # --- Contraction: C = A_(k) @ B_(k)^T ---
    C = A_unfolded * B_unfolded.ctranspose()

    return C.double()