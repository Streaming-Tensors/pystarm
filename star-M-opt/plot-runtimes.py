import pystarm
import numpy as np
import pyttb as ttb
import math
import matplotlib.pyplot as plt
import time
import cotengra as ctg
import string
import os
from datetime import datetime

def make_plot(x, y, logscale_y=False, xlabel='', ylabel='', title='', plot_file='figures/plot.svg'):
    plt.plot(x, y, '-o')
    plt.xticks(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logscale_y: plt.yscale('log')
    plt.title(title)
    plt.savefig(plot_file, dpi=1200)
    print(f"\nDONE! Plot can be found at {plot_file}")


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

def contegra_tensor_contract_timed(A, B, k, optimize="auto-hq"):
    """
    Compute A_(k) @ B_(k)^T, the mode-k unfolding contraction of two
    tensors, without explicitly unfolding either tensor.

    Parameters
    ----------
    A, B : array_like
        Tensors of the same number of dimensions. All dimensions must
        match except possibly dimension `k`.
    k : int
        The mode to keep free (the unfolding mode). Can be negative
        (numpy-style indexing).
    optimize : str or cotengra.PathOptimizer, default "auto-hq"
        Passed straight through to cotengra. Use "auto-hq" for cotengra's
        high-quality hyper-optimized path (good default for most sizes),
        or pass an explicit ctg.HyperOptimizer(...) instance for large
        / performance-critical tensors where you want to tune the search.

    Returns
    -------
    ndarray of shape (A.shape[k], B.shape[k])
        Equal to `unfold(A, k) @ unfold(B, k).T`.
    """
    A = np.asarray(A) if not hasattr(A, "shape") else A
    B = np.asarray(B) if not hasattr(B, "shape") else B

    if A.ndim != B.ndim:
        raise ValueError(
            f"A and B must have the same number of dimensions "
            f"(got {A.ndim} and {B.ndim})."
        )

    ndim = A.ndim
    k = k % ndim  # normalize negative index

    for ax in range(ndim):
        if ax == k:
            continue
        if A.shape[ax] != B.shape[ax]:
            raise ValueError(
                f"A and B must match in every dimension except k={k}, "
                f"but differ at dim {ax}: {A.shape[ax]} vs {B.shape[ax]}."
            )

    # Assign a distinct einsum letter to every shared (summed) mode,
    # and separate letters for the two free mode-k axes.
    letters = iter(string.ascii_lowercase + string.ascii_uppercase)
    shared = [next(letters) for _ in range(ndim - 1)]

    n_k = next(letters)
    p = next(letters)

    subs_A, subs_B = [], []
    it = iter(shared)
    for ax in range(ndim):
        if ax == k:
            subs_A.append(n_k)
            subs_B.append(p)
        else:
            c = next(it)
            subs_A.append(c)
            subs_B.append(c)

    eq = f"{''.join(subs_A)},{''.join(subs_B)}->{n_k}{p}"

    t0 = time.perf_counter()
    C = ctg.einsum(eq, A, B, optimize=optimize)
    t1 = time.perf_counter()
    print(f"Elapsed time cotengra = {t1-t0} seconds")
    return C,  t1-t0

def tensor_contract_batched(A_np, B_np, k, n_threads=3):
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
    start = time.perf_counter()
    C = pystarm.tensor_contract_all_but_one_batched(A, B, k, n_threads)
    end = time.perf_counter()
    print(f"Elapsed time batched (nthreads={n_threads}) = {end - start} seconds")
    # Return to numpy
    C_np = np.frombuffer(C, dtype=np.float64).reshape(C.getdims(), order='F', copy = False)
    
    return C_np, end - start

def plot_runtimes_naive_ttb_numpy(num_iter_per_n = 10, n_values = [50,75,100,125,150]):
    '''
    This is the code used to generate the plots used in my poster.
    '''
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

def plot_runtimes_naive_reduction(num_iter_per_n=10, n_values=[50,100,150,200,250]):
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
    
    plt.plot(n_values, naive_times, '-o')
    plt.plot(n_values, reduction_times, '-o')
    plt.xticks(n_values)
    plt.xlabel("n")
    plt.ylabel("Average time to compute contraction (seconds)")
    plt.yscale('log')
    plt.title(f"Comparing compute time of tensor contractions, {num_iter_per_n} iterations per n")
    plt.legend(["Naive parallel contraction", "OpenMP reduction contraction"])
    plt.savefig("naive-vs-reduction-contraction-runtimes-10-per-n.svg", dpi=1200)

def plot_runtimes_omp_batched_cotengra(
        num_iter_per_n, 
        n_values=[25,50,100,150,200],
        plot_file="contraction-runtimes-omp-vs-batched-vs-cotengra.svg",
        n_threads_batched=3
    ):

    print(
        f"COMPARING RUNTIMES BETWEEN OMP REDUCTION, BATCHED BLAS, AND COTENGRA EINSUM...",
        f"NUM_ITER_PER_N = {num_iter_per_n}\nn VALUES TESTING: {n_values}"
        )
    j = 0

    omp_times = np.zeros(len(n_values))
    batched_times = np.zeros(len(n_values))
    cotengra_times = np.zeros(len(n_values))
    for n in n_values:
        print(f"COMPARING RUNTIMES FOR TENSORS OF SIZE {n}x{n}x{n}x{n} {num_iter_per_n} TIMES...")
        for _ in range(num_iter_per_n):
            A_dims = (n,n,n,n)
            A_nelm = math.prod(A_dims)

            B_dims = (n,n,n,n)
            B_nelm = math.prod(B_dims)

            # Create numpy tensors
            A_np = np.asfortranarray(np.random.rand(A_nelm).reshape(A_dims, order='F'))
            B_np = np.asfortranarray(np.random.rand(B_nelm).reshape(B_dims, order='F'))

            # Compare results for ALL modes to check for an indexing offset
            k = 2
            C_omp, omp_time = tensor_contract_multiply(A_np, B_np, k, naive=False)
            del C_omp
            C_batched, batched_time = tensor_contract_batched(A_np, B_np, k, n_threads_batched)
            del C_batched
            C_cotengra, cotengra_time = contegra_tensor_contract_timed(A_np, B_np, k)
            del C_cotengra
            omp_times[j] += omp_time
            batched_times[j] += batched_time
            cotengra_times[j] += cotengra_time

        
        omp_times[j] /= num_iter_per_n
        batched_times[j] /= num_iter_per_n
        cotengra_times[j] /= num_iter_per_n
        j +=1 

    n_vals_string = '_'.join(str(n) for n in n_values)
    cwd = os.getcwd()
    now = datetime.now()
    date_formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
    omp_file = f"{cwd}/data/{date_formatted}_omp_times_{n_vals_string}.npy"
    batched_file = f"{cwd}/data/{date_formatted}_batched_times_{n_vals_string}.npy"
    cotengra_file = f"{cwd}/data/{date_formatted}_cotengra_times_{n_vals_string}.npy"

    
    np.save(omp_file, omp_times)
    np.save(batched_file, batched_times)
    np.save(cotengra_file, cotengra_times)
    plt.plot(n_values, omp_times, '-o')
    plt.plot(n_values, batched_times, '-o')
    plt.plot(n_values, cotengra_times, '-o')
    plt.xticks(n_values)
    plt.xlabel("n")
    plt.ylabel("Average time to compute contraction (seconds)")
    plt.yscale('log')
    plt.title(f"Comparing compute time of tensor contractions, {num_iter_per_n} iterations per n")
    plt.legend(["OMP Reduction", f"batched dgemm\n(n_threads={n_threads_batched})", "cotengra einsum"])
    plot_filepath = f"{cwd}/figures/{date_formatted}_{plot_file}"
    plt.savefig(plot_filepath, dpi=1200)
    print(
        f"\nDONE! Plot can be found at {plot_filepath}",
        f"\nNumpy files with the runtimes can be found in {omp_file}, {batched_file}, and {cotengra_file}."
    )

def plot_runtimes_nthreads_in_batched_contraction(
        num_iter_per_n=10,
        nthreads_vals_to_test=[3, 6, 9, 12, 15],
        tensor_dims=(50,50,50,50),
        plot_file="batched_contraction_threads_runtime_comparison"
        ):
    print(
        f"COMPARING RUNTIMES BETWEEN THREAD SIZES IN BATCHED TENSOR CONTRACTION IMPLEMENTATION.",
        f"NUM_ITER_PER_N = {num_iter_per_n}\nn VALUES TESTING: {nthreads_vals_to_test}"
        )
    j = 0

    runtimes = np.zeros(len(nthreads_vals_to_test))
    nelm = np.prod(tensor_dims)
                
    # Create numpy tensors
    A_np = np.asfortranarray(np.random.rand(nelm).reshape(tensor_dims, order='F'))
    B_np = np.asfortranarray(np.random.rand(nelm).reshape(tensor_dims, order='F'))

    for n in nthreads_vals_to_test:
        print(f"COMPARING RUNTIMES FOR TENSORS OF SIZE {tensor_dims} {num_iter_per_n} TIMES...")
        for _ in range(num_iter_per_n):
            # Compare results for ALL modes to check for an indexing offset
            k = 2
            C_batched, batched_time = tensor_contract_batched(A_np, B_np, k, n)
            del C_batched
            runtimes[j] += batched_time
        
        runtimes[j] /= num_iter_per_n
        j +=1 

    n_vals_string = '_'.join(str(n) for n in nthreads_vals_to_test)
    cwd = os.getcwd()
    now = datetime.now()
    date_formatted = now.strftime("%Y-%m-%d_%H:%M:%S")

    batched_file = f"{cwd}/data/{date_formatted}_batched_times_nthreads_{n_vals_string}.npy"
    
    np.save(arr=runtimes, file=batched_file)

    make_plot(nthreads_vals_to_test, runtimes, logscale_y=False, xlabel='nthreads',
          ylabel=f"time (seconds), {num_iter_per_n} contractions per x.",
          title=f"Average time to compute batched contraction on {tensor_dims} tensor",
          plot_file=f"{cwd}/figures/{date_formatted}_{plot_file}.svg")
    
    print(f"\nNumpy file with the runtimes can be found in {batched_file}.")

plot_runtimes_omp_batched_cotengra(10, n_threads_batched=12)
