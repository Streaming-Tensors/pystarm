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
np.random.seed(1234)


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

    # y = np.random.rand(x.size).reshape(x.shape, order='F')
    # x2 = x
    # x = y
    t0 = time.perf_counter()
    A = pystarm.Tensor(x, len(x.shape), x.shape)
    t1 = time.perf_counter()
    print("Time to convert to Pystarm tensor:", t1-t0)
    
    A_hat = None
    Ms = []
    MTs = []
    # ttm_modes = [2,3]
    # ttm_modes = [4,3,2]
    ttm_modes = [2,3,4]
    # ttm_modes = [2]
    
    Adims = [A.getdims()[mode] for mode in ttm_modes]
    DFs = [np.asfortranarray(dct(np.eye(n, dtype=np.float64), axis=0, norm="ortho")) for n in Adims]
    DFTs = [np.asfortranarray(DF.T) for DF in DFs]
    Ms = [pystarm.Matrix(DF, DF.shape[0], DF.shape[1]) for DF in DFs]
    MTs = [pystarm.Matrix(DFT, DFT.shape[0], DFT.shape[1]) for DFT in DFTs]

    print((A.norm() - np.linalg.norm(x)) / A.norm())
    
    # Ms[0].print()

    A_hat = pystarm.transform(A, Ms, ttm_modes)
    print(A_hat.norm())
    A_tilde = pystarm.transform(A_hat, MTs, ttm_modes)
    print(A_tilde.norm())

    # A1 = pystarm.ttm(A, Ms[0], 2)
    # print(A1.norm())
    # A_hat = pystarm.ttm(A1, Ms[1], 3)
    # print(A_hat.norm())

    # A2 = pystarm.ttm(A_hat, MTs[0], 2)
    # print(A2.norm())
    # A_tilde = pystarm.ttm(A2, MTs[1], 3)
    # print(A_tilde.norm())

    # U_hat, S_hat, VT_hat = pystarm.tsvdmi_compress(A, Ms, k)
    # A_tilde = pystarm.tsvdmi_reconstruct(U_hat, S_hat, VT_hat, MTs)
    
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

# # import pystarm
# # pystarm.check()
