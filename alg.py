import time
import numpy as np
import scipy as sp
from scipy.fft import dct
import numpy as np
import matplotlib.pyplot as plt
import pystarm

def tsvdm_I_compress(A, M, k):
    t0 = time.perf_counter()
    A_hat = pystarm.ttm(A, M, len(A.getdims())-1) 
    t1 = time.perf_counter()
    print("[tsvdm_I_compress] Time for TTM A x M:", t1-t0)

    t0 = time.perf_counter()
    U_hat, S_hat, V_hat = pystarm.slicewise_svdx(A_hat, k)
    # U_hat, S_hat, V_hat = pystarm.slicewise_svd(A_hat)
    t1 = time.perf_counter()
    print("[tsvdm_I_compress] Time for slicewise SVD:", t1-t0)
    
    return (U_hat,S_hat,V_hat)

def tsvdm_I_compress_dway(A, listM, k):
    A_hat = None
    for x in listM:
        t0 = time.perf_counter()
        A_hat = pystarm.ttm(A, M, len(A.getdims())-1) 
        t1 = time.perf_counter()
        print("[tsvdm_I_compress] Time for TTM A x M:", t1-t0)

    t0 = time.perf_counter()
    U_hat, S_hat, V_hat = pystarm.slicewise_svdx(A_hat, k)
    # U_hat, S_hat, V_hat = pystarm.slicewise_svd(A_hat)
    t1 = time.perf_counter()
    print("[tsvdm_I_compress] Time for slicewise SVD:", t1-t0)
    
    return (U_hat,S_hat,V_hat)

def tsvdm_I_reconstruct(U_hat, S_hat, VT_hat, Minv):
    t0 = time.perf_counter()
    A_hat = pystarm.slicewise_matmul(U_hat, S_hat, VT_hat)
    t1 = time.perf_counter()
    print("[tsvdm_I_reconstruct] Time to reconstruct A_hat:", t1-t0)

    t0 = time.perf_counter()
    Atilde = pystarm.ttm(A_hat, Minv, len(A_hat.getdims())-1) 
    t1 = time.perf_counter()
    print("[tsvdm_I_reconstruct] Time for TTM A_hat x Minv:", t1-t0)
    
    return Atilde
