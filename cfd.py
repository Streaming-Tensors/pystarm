import time
import pystarm
import numpy as np
import scipy as sp
from scipy.fft import dct
import numpy as np
import matplotlib.pyplot as plt
import h5py
from alg import tsvdm_I_compress
from alg import tsvdm_I_reconstruct
import sys
import pyttb as ttb


if __name__ == "__main__":
    k = int(sys.argv[1])
    t0 = time.perf_counter()
    file_dir = "/global/cfs/cdirs/m4293/starMData/"
    file_prefix = "tgv_"
    frames = []
    for snapshot in ["0000", "0001", "0002", "0003", "0004", "0005"]:
        file_name = file_dir + file_prefix + snapshot + ".h5"
        print("Reading", file_name)
        with h5py.File(file_name, 'r') as f:
            # # List all groups and datasets in the file
            # print("Keys in the file:", list(f.keys()))
            dset = f['tensor']
            frames.append(np.array(dset))
    full_array = np.stack(frames, axis=0)
    full_array = np.transpose(full_array, (1, 2, 3, 4, 0)) # Push the time dimension (0th dim) to the last mode (contiguous in F order)
    # Make explicit copy of the permuted data to take ownership of the associated memory
    x = np.zeros(full_array.shape, dtype=np.float64, order='F')
    for i in range(full_array.shape[-1]):
        x[...,i] = full_array[...,i]
    # print(x.flags)
    # print(x.shape)
    t1 = time.perf_counter()
    print("Time to read data into numpy:", t1-t0)
    print("Tensor shape:", x.shape) 

    t0 = time.perf_counter()
    A = pystarm.Tensor(x, len(x.shape), x.shape)
    t1 = time.perf_counter()
    print("Time to convert to Pystarm tensor:", t1-t0)
    
    A_hat = None
    Ms = []
    MTs = []
    # ttm_modes = [2,3,4]
    ttm_modes = [4,3,2]
    # ttm_modes = [1]
    for i in range(len(ttm_modes)):
        # print("[", i, "]")
        mode = ttm_modes[i]
        n = A.getdims()[mode]
        print("Dimension in mode", mode, "is", n)

        t0 = time.perf_counter()
        mat_nelm = n * n
        mat_dims = (n, n)
        mat_ndim = len(mat_dims)
        DC = dct(np.eye(n, dtype=np.float64), axis=0, norm="ortho")
        # DC, _ = np.linalg.qr(np.random.randn(n, n))
        DF = np.asfortranarray(DC)
        DFT = np.asfortranarray(DF.T)

        # print(np.matmul(DF, DFT))

        M = pystarm.Matrix(DF, mat_dims[0], mat_dims[1])
        MT = pystarm.Matrix(DFT, mat_dims[0], mat_dims[1])
        t1 = time.perf_counter()
        print("Time to generate the DCT matrix of dim", n, ":", t1-t0)

        Ms.append(M)
        MTs.append(MT)
        
        t0 = time.perf_counter()
        if i == 0:
            A_hat_temp = pystarm.ttm(A, Ms[i], mode) 
            # A_hat_temp = pystarm.ttm_loop(A, Ms[i], mode) 
        else:
            A_hat_temp = pystarm.ttm(A_hat, Ms[i], mode) 
            # A_hat_temp = pystarm.ttm_loop(A_hat, Ms[i], mode) 
        t1 = time.perf_counter()
        y = np.frombuffer(A_hat_temp, dtype=np.float64).reshape(A_hat_temp.getdims(), order='F', copy = False)
        print(np.linalg.norm(y))
        if A_hat is not None:
            # To prevent calling clear in the first iteration
            # Memory would be cleared in the subsequent iterations, as new memory is allocated with new TTM
            A_hat.clear()
        A_hat = A_hat_temp
        print("Time for TTM on mode", mode, ":", t1-t0)

    # ### Test against pyttb multi-ttm
    # ttb_Ms = []
    # for M in Ms:
        # ttb_M = np.frombuffer(M, dtype=np.float64).reshape(M.getdims(), order='F', copy = False)
        # ttb_Ms.append(ttb_M)
    # ttb_A = ttb.tensor(full_array, copy=True)
    # ttb_A_hat = ttb_A.ttm(ttb_Ms, ttm_modes)
    # ttb_A_hat = ttb_A_hat.data
    # np_A_hat = np.frombuffer(A_hat, dtype=np.float64).reshape(A_hat.getdims(), order='F', copy = False)
    # A_hat_diff = ttb_A_hat - np_A_hat
    # print(A_hat_diff)
    # norm_A_hat_diff = np.linalg.norm(A_hat_diff)
    # print("TTM difference:", norm_A_hat_diff)

    # print("ttb A_hat:", ttb_A_hat)
    # print("np A_hat:", np_A_hat)

    t0 = time.perf_counter()
    U_hat, S_hat, VT_hat = pystarm.slicewise_svdx(A_hat, k)
    t1 = time.perf_counter()
    print("Time for slicewise SVD:", t1-t0)
    if A_hat is not None:
        A_hat.clear()

    print("S_hat:", S_hat.getdims() )
    print("S_hat:", S_hat.getcol(0) )
    # S_hat.print()

    t0 = time.perf_counter()
    A_hat = pystarm.slicewise_matmul(U_hat, S_hat, VT_hat)
    t1 = time.perf_counter()
    print("Time to reconstruct A_hat:", t1-t0)

    y = np.frombuffer(A_hat, dtype=np.float64).reshape(A_hat.getdims(), order='F', copy = False)
    print("norm(A_hat) after SVD:", np.linalg.norm(y))

    A_tilde = None
    # for i in reversed(range(len(ttm_modes))):
    for i in range(len(ttm_modes)):
        # print("[", i, "]")
        mode = ttm_modes[i]
        n = A_hat.getdims()[mode]
        print("Dimension in mode", mode, "is", n)

        # t0 = time.perf_counter()
        # mat_nelm = n * n
        # mat_dims = (n, n)
        # mat_ndim = len(mat_dims)
        # DC = dct(np.eye(n), axis=0, norm="ortho")
        # DF = np.asfortranarray(DC)
        # DFT = np.asfortranarray(DF.T)

        # M = pystarm.Matrix(DF, mat_dims[0], mat_dims[1])
        # MT = pystarm.Matrix(DFT, mat_dims[0], mat_dims[1])
        # t1 = time.perf_counter()
        # print("Time to generate the DCT matrix of dim", n, ":", t1-t0)
        
        t0 = time.perf_counter()
        if i == 0:
            A_tilde_temp = pystarm.ttm(A_hat, MTs[i], mode) 
            # A_tilde_temp = pystarm.ttm_loop(A_hat, MTs[i], mode) 
        else:
            A_tilde_temp = pystarm.ttm(A_tilde, MTs[i], mode) 
            # A_tilde_temp = pystarm.ttm_loop(A_tilde, MTs[i], mode) 
        t1 = time.perf_counter()
        y = np.frombuffer(A_tilde_temp, dtype=np.float64).reshape(A_tilde_temp.getdims(), order='F', copy = False)
        print(np.linalg.norm(y))
        if A_tilde is not None:
            A_tilde.clear()
        A_tilde = A_tilde_temp
        print("Time for TTM on mode", mode, ":", t1-t0)

    x_reconst = np.frombuffer(A_tilde, dtype=np.float64).reshape(A_tilde.getdims(), order='F', copy = False)
    x_diff = x - x_reconst
    norm_x_diff = np.linalg.norm(x_diff)
    norm_x = np.linalg.norm(x)
    norm_x_reconst = np.linalg.norm(x_reconst)

    # print(x_diff)

    print("Absolute err:", norm_x_diff)
    print("Relative err:", norm_x_diff/norm_x)
    print("Norm of original array:", norm_x)
    print("Norm of reconstructed array:", norm_x_reconst)


