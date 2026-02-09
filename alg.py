import time
import numpy as np
import scipy as sp
from scipy.fft import dct
import numpy as np
import matplotlib.pyplot as plt
import pystarm

def tsvdm_I_compress(A, Ms, ttm_modes, k, verbose=False):
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

    t0 = time.perf_counter()
    U_hat, S_hat, V_hat = pystarm.slicewise_svdx(A_hat, k)
    # U_hat, S_hat, V_hat = pystarm.slicewise_svd(A_hat)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_I_compress]", "Time for slicewise SVD:", t1-t0)
    
    return (U_hat,S_hat,V_hat)

def tsvdm_I_reconstruct(U_hat, S_hat, VT_hat, Minvs, ttm_modes, verbose=False):
    t0 = time.perf_counter()
    A_hat = pystarm.slicewise_matmul(U_hat, S_hat, VT_hat)
    t1 = time.perf_counter()
    if verbose:
        print("[tsvdm_I_reconstruct] Time to reconstruct A_hat:", t1-t0)

    if verbose:
        print("[tsvdm_I_reconstruct]", "ttm_modes", ttm_modes)

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
    
    return A_tilde
