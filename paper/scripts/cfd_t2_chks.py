import time
import numpy as np
import scipy as sp
from scipy.fft import dct
import h5py
import sys
import argparse
import pystarm

def read_cfd_data(porder=[1, 2, 3, 4, 0]):
    file_dir = "/home/seswar/starM-HPC-Updated/starMData/"
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
    full_array = np.transpose(full_array, tuple(porder)) # Push the time dimension (0th dim) to the last mode (contiguous in F order)
    # Make explicit copy of the permuted data to take ownership of the associated memory
    x = np.zeros(full_array.shape, dtype=np.float64, order='F')
    for i in range(full_array.shape[-1]):
        x[...,i] = full_array[...,i]

    return x

def read_cfd_data2(porder=[1, 2, 3, 4, 0]):
    file_dir = "/home/seswar/starM-HPC-Updated/starMData/"
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
    full_array = np.transpose(full_array, tuple(porder)) # Push the time dimension (0th dim) to the last mode (contiguous in F order)
    # Make explicit copy of the permuted data to take ownership of the associated memory
    x = np.zeros(full_array.shape, dtype=np.float64, order='F')
    for m in range(x.shape[4]):
        for l in range(x.shape[3]):
            for k in range(x.shape[2]):
                for j in range(x.shape[1]):
                    for i in range(x.shape[0]):
                        x[i, j, k, l, m] = full_array[i, j, k, l, m]
    return x

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
    parser.add_argument("-t", "--tol", type=float, help="Error tolerance for tsvdm-II")
    args = parser.parse_args()

    tol = 0.01
    if args.tol is not None:
        tol = args.tol

    porder = [1, 2, 3, 4, 0] # Push the time dimension to the last mode
    print(f"CFD checks with order:\n order: {porder} \n tol  : {tol}")    

    t0 = time.perf_counter()
    x  = read_cfd_data(porder)
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
