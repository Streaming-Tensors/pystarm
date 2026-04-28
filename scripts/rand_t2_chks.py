import time
import numpy as np
import scipy as sp
from scipy.fft import dct
import h5py
import sys
import argparse
import pystarm

def tsvdm_II_tf(A, thr, verbose=False):
    t0 = time.perf_counter()
    U, S, Vt = pystarm.slicewise_svd(A)
    t1 = time.perf_counter()
    print(f"[tsvdm_II_tf] Time for full SVD: {t1-t0}")

    t0 = time.perf_counter()
    ks = pystarm.thresholds(S, thr)
    t1 = time.perf_counter()
    print(f"[tsvdm_II_tf] Time for threshold computation: {t1-t0}")

    t0 = time.perf_counter()
    Uk, Sk, Vkt = pystarm.truncate_factors(U, S, Vt, ks)
    t1 = time.perf_counter()
    print(f"[tsvdm_II_tf] Time for truncation: {t1-t0}")

    # Clear tensors
    U.clear()
    S.clear()
    Vt.clear()

    return Uk, Sk, Vkt

def tsvdm_II_std(A, thr, verbose=False):
    t0 = time.perf_counter()
    S = pystarm.slicewise_svdvals(A)
    t1 = time.perf_counter()
    print(f"[tsvdm_II_std] Time for svdvals: {t1-t0}")

    t0 = time.perf_counter()
    ks = pystarm.thresholds(S, thr)
    t1 = time.perf_counter()
    print(f"[tsvdm_II_std] Time for threshold computation: {t1-t0}")

    t0 = time.perf_counter()
    Uk, Sk, Vkt = pystarm.slicewise_svdks(A, ks)
    t1 = time.perf_counter()
    print(f"[tsvdm_II_std] Time for SVD ks: {t1-t0}")

    # Clear stuff
    S.clear()

    return Uk, Sk, Vkt

def tsvdm_II_sc(A, thr, verbose=False):
    t0 = time.perf_counter()
    Uk, Sk, Vkt = pystarm.slicewise_svd_thr(A, thr)
    t1 = time.perf_counter()
    print(f"[tsvdm_II_sc] Time for SVD thr: {t1-t0}")

    return Uk, Sk, Vkt

def cmp_factors(ref_factors, factors):
    Uref, Sref, Vtref = ref_factors
    Ucmp, Scmp, Vtcmp = factors     

    nslices = Sref.ncol
    Aflags  = np.zeros(nslices, dtype=np.bool)
    Sflags  = np.zeros(nslices, dtype=np.bool)

    for ii in range(nslices):
        Ur  = Uref.getfrontalslice(ii)
        Urp = np.frombuffer(Ur, dtype=np.float64).reshape(Ur.getdims(), order='F', copy=False)
        Uc  = Ucmp.getfrontalslice(ii)
        Ucp = np.frombuffer(Uc, dtype=np.float64).reshape(Uc.getdims(), order='F', copy=False)

        Vtr  = Vtref.getfrontalslice(ii)
        Vtrp = np.frombuffer(Vtr, dtype=np.float64).reshape(Vtr.getdims(), order='F', copy=False)
        Vtc  = Vtcmp.getfrontalslice(ii)
        Vtcp = np.frombuffer(Vtc, dtype=np.float64).reshape(Vtc.getdims(), order='F', copy=False)

        sr   = np.array(Sref.getcol(ii))
        sc   = np.array(Scmp.getcol(ii))

        Aflags[ii] = np.allclose(Urp @ np.diag(sr) @ Vtrp, Ucp @ np.diag(sc) @ Vtcp)
        Sflags[ii] = np.allclose(sr, sc)

    return Aflags, Sflags
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tol", type=float, default=0.01, help="Error tolerance for tsvdm-II")
    parser.add_argument("-d", "--dims", type=int, nargs="*", default=[5, 4, 3], help="Dimensions of the random tensor")
    args = parser.parse_args()

    tol  = args.tol
    dims = args.dims

    print(f"Working on random tensor:\n dims: {dims} \n tol: {tol}")    

    t0 = time.perf_counter()
    x  = np.random.randn(np.prod(dims)).reshape(tuple(dims), order="F") 
    t1 = time.perf_counter()

    print("Time to read data into numpy:", t1-t0)
    print("Tensor shape:", x.shape)

    ten = pystarm.Tensor(x, len(x.shape), x.shape)

    t0 = time.perf_counter()
    U1, S1, V1t = tsvdm_II_tf(ten, tol)
    t1 = time.perf_counter()
    print(f"tsvdm_II_tf func call: {t1-t0}")

    t0 = time.perf_counter()
    U2, S2, V2t = tsvdm_II_std(ten, tol)
    t1 = time.perf_counter()
    print(f"tsvdm_II_std func call: {t1-t0}")

    t0 = time.perf_counter()
    U3, S3, V3t = tsvdm_II_sc(ten, tol)
    t1 = time.perf_counter()
    print(f"tsvdm_II_sc func call: {t1-t0}")

    # Check all factors against the tsvdm_II_std
    print("Checking truncation.")
    tf_aflags, tf_sflags = cmp_factors((U2, S2, V2t), (U1, S1, V1t))
    print(f"A flags : {np.all(tf_aflags)}")
    print(f"S flags : {np.all(tf_sflags)}")

    print("Checking saving computations.")
    sc_aflags, sc_sflags = cmp_factors((U2, S2, V2t), (U3, S3, V3t))
    print(f"A flags : {np.all(sc_aflags)}")
    print(f"S flags : {np.all(sc_sflags)}")

    # Clear all data
    U1.clear()
    S1.clear()
    V1t.clear()

    U2.clear()
    S2.clear()
    V2t.clear()

    U3.clear()
    S3.clear()
    V3t.clear()
