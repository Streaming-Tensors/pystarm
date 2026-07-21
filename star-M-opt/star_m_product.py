import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pystarm
import numpy as np
import pyttb as ttb
import math
import matplotlib.pyplot as plt
import time

def tensor_contract_multiply(A_np, B_np, k, naive: bool):

    A_ndim = A_np.ndim
    B_ndim = B_np.ndim

    assert A_ndim >= 3, "A must be a tensor with ndim >= 3"
    assert B_ndim >= 3, "B must be a tensor with ndim >= 3"
    
    A_shape = A_np.shape
    B_shape = B_np.shape

    # Ensure dimensions are correct.
    assert A_shape[A_ndim - 1] == B_shape[B_ndim - 1], "A and B must have the same number of slices."

    A = pystarm.Tensor(A_np, A_ndim, A_shape)
    B = pystarm.Tensor(B_np, B_ndim, B_shape)
    
    # Slicewise multiply A and B
    start = time.perf_counter()
    C = pystarm.tensor_contract_all_but_one(A, B, k, naive)
    end = time.perf_counter()
    print(f"Elapsed time naive = {naive} PYSTARM: {end - start} seconds")
    # Return to numpy
    C_np = np.frombuffer(C, dtype=np.float64).reshape(C.getdims(), order='F', copy = False)
    
    return C_np, end-start

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
    if A.shape != B.shape:
        raise ValueError(
            f"Tensors must have the same shape. Got A.shape={A.shape}, B.shape={B.shape}"
        )

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
    start = time.perf_counter()
    A_unfolded = A_ttb.to_tenmat(rdims=np.array([mode]))
    del A_ttb
    B_unfolded = B_ttb.to_tenmat(rdims=np.array([mode]))
    del B_ttb

    # --- Contraction: C = A_(k) @ B_(k)^T ---
    C = A_unfolded * B_unfolded.ctranspose()
    end = time.perf_counter()
    print(f"Elapsed time TTB: {end - start} seconds")

    return C.double(), end-start

def np_einsum_contract(A_np, B_np):
    '''
    This only works for mode = 2

    '''
    start = time.perf_counter()
    C = np.einsum("ijkl,ijdl->kd", A_np, B_np)
    end = time.perf_counter()
    print(f"Elapsed time numpy: {end - start} seconds")
    return C, end - start

def plot_runtimes_naive_ttb_numpy():
    '''
    This is the code used to generate the plots used in my poster.
    '''
    num_iter_per_n = 5 # Number of iterations per n. All tensors will be of dimensions n x n x n x n.
    n_values = [50,75,100,125,150]
    i = 0
    j = 0

    cpp_times = np.zeros(len(n_values))
    ttb_times = np.zeros(len(n_values))
    np_times = np.zeros(len(n_values))
    for n in n_values:
        for i in range(num_iter_per_n):
            A_dims = (n,n,n,n)
            A_nelm = math.prod(A_dims)
            A_ndim = len(A_dims)

            B_dims = (n,n,n,n)
            B_nelm = math.prod(B_dims)
            B_ndim = len(B_dims)

            # Create numpy tensors
            A_np = np.asfortranarray(np.random.rand(A_nelm).reshape(A_dims, order='F'))
            B_np = np.asfortranarray(np.random.rand(B_nelm).reshape(B_dims, order='F'))

            # Compare results for ALL modes to check for an indexing offset
            k = 2
            C_cpp, cpp_time = tensor_contract_multiply(A_np, B_np, k)
            del C_cpp
            C_ttb, ttb_time = tensor_contraction_product_ttb(A_np, B_np, mode=k)
            del C_ttb
            C_np, np_time = np_einsum_contract(A_np, B_np)
            del C_np
            cpp_times[j] += cpp_time
            ttb_times[j] += ttb_time
            np_times[j] += np_time

        
        cpp_times[j] /= num_iter_per_n
        ttb_times[j] /= num_iter_per_n
        np_times[j] /= num_iter_per_n
        j +=1 

    np.save("data/cpp_times", cpp_times)
    np.save("data/ttb_times", ttb_times)
    np.save("data/np_times", np_times)
    plt.plot(n_values, cpp_times, '-o')
    plt.plot(n_values, ttb_times, '-o')
    plt.plot(n_values, np_times, '-o')
    plt.xticks(n_values)
    plt.xlabel("n")
    plt.ylabel("Average time to compute contraction (seconds)")
    plt.yscale('log')
    plt.title("Comparing compute time of tensor contractions, 5 iterations per n")
    plt.legend(["C++ pystarm kernel", "pyttb contraction", "np.einsum"])
    plt.savefig("figures/contraction-runtimes.svg", dpi=1200)

def plot_runtimes_naive_reduction():
    num_iter_per_n = 5 # Number of iterations per n. All tensors will be of dimensions n x n x n x n.
    n_values = [50,100,150,200,250]
    i = 0
    j = 0

    naive_times = np.zeros(len(n_values))
    reduction_times = np.zeros(len(n_values))
    for n in n_values:
        for i in range(num_iter_per_n):
            A_dims = (n,n,n,n)
            A_nelm = math.prod(A_dims)

            B_dims = (n,n,n,n)
            B_nelm = math.prod(B_dims)

            # Create numpy tensors
            A_np = np.asfortranarray(np.random.rand(A_nelm).reshape(A_dims, order='F'))
            B_np = np.asfortranarray(np.random.rand(B_nelm).reshape(B_dims, order='F'))

            # Compare results for ALL modes to check for an indexing offset
            k = 2
            C_naive, naive_time = tensor_contract_multiply(A_np, B_np, k, True)
            del C_naive
            C_reduction, reduction_time = tensor_contract_multiply(A_np, B_np, k, False)
            del C_reduction
            naive_times[j] += naive_time
            reduction_times[j] += reduction_time

        
        naive_times[j] /= num_iter_per_n
        reduction_times[j] /= num_iter_per_n
        j +=1 
    
    np.save("data/naive_times", naive_times)
    np.save("data/reduction_times", reduction_times)
    plt.plot(n_values, naive_times, '-o')
    plt.plot(n_values, reduction_times, '-o')
    plt.xticks(n_values)
    plt.xlabel("n")
    plt.ylabel("Average time to compute contraction (seconds)")
    plt.yscale('log')
    plt.title(f"Comparing compute time of tensor contractions, {num_iter_per_n} iterations per n")
    plt.legend(["Naive parallel contraction", "OpenMP reduction contraction"])
    plt.savefig("figures/naive-vs-reduction-contraction-runtimes.svg", dpi=1200)

plot_runtimes_naive_reduction()