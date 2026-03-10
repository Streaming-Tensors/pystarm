import argparse
import os
import time
import numpy as np
import h5py
import cv2
from scipy.fft import dct
import pystarm
import pyttb as ttb
from alg import tsvdm_I_compress, tsvdm_I_reconstruct
from alg import tsvdm_II_compress, tsvdm_II_reconstruct


def read_soccer_data(filepath):
    """Read a grayscale soccer video (.mp4) and return an H x W x T Fortran-order float64 tensor."""
    cap = cv2.VideoCapture(filepath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # H x W
        frames.append(gray)
    cap.release()

    t_h_w = np.stack(frames, axis=0)  # T x H x W
    h_w_t = np.transpose(t_h_w, (1, 2, 0))  # H x W x T

    x = np.zeros(h_w_t.shape, dtype=np.float64, order='F')
    for i in range(h_w_t.shape[2]):
        x[:, :, i] = h_w_t[:, :, i]
    return x


def read_traffic_data(filepath):
    """Read a traffic binary file (.bin) and return an H x W x 3 x T Fortran-order float64 tensor.

    The binary format is defined by scripts/traffic_writer.m which wrote this file.
    Header layout: 4 bytes uint32 height, 4 bytes uint32 width, 8 bytes float64 fps.
    The rest of the file is the frame data written as float64 values.
    """
    with open(filepath, 'rb') as f:
        height = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        width  = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        _fps   = np.frombuffer(f.read(8), dtype=np.float64)[0]  # not needed for experiments
        data   = np.frombuffer(f.read(), dtype=np.float64)

    # Number of frames is not stored in the header; derived by dividing total values by one frame size (H x W x 3).
    frame_size = height * width * 3
    num_frames = len(data) // frame_size

    # MATLAB serializes arrays in column-major (Fortran) order by default, so we
    # reshape with order='F' to match. The 3 corresponds to RGB channels, as
    # MATLAB's readFrame() returns each frame as H x W x 3.
    return data.reshape((height, width, 3, num_frames), order='F')


def read_cfd_data(dirpath):
    """Read CFD snapshots from a directory of HDF5 files (.h5) and return a Fortran-order float64 tensor.

    Each file is named tgv_<snapshot>.h5 and contains a dataset named 'tensor'.
    Snapshots are stacked along a new time axis, which is pushed to the last mode.
    """
    file_prefix = "tgv_"
    snapshots = ["0000", "0001", "0002", "0003", "0004", "0005"]

    frames = []
    for snapshot in snapshots:
        filepath = os.path.join(dirpath, file_prefix + snapshot + ".h5")
        with h5py.File(filepath, 'r') as f:
            frames.append(np.array(f['tensor']))

    # Stack along axis 0 giving T x ... then push time to last mode
    full_array = np.stack(frames, axis=0)
    ndim = full_array.ndim
    perm = tuple(range(1, ndim)) + (0,)  # push time axis (0) to last
    full_array = np.transpose(full_array, perm)

    x = np.zeros(full_array.shape, dtype=np.float64, order='F')
    for i in range(full_array.shape[-1]):
        x[..., i] = full_array[..., i]
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-alg",       "--alg",       type=str,   help="Name of the algorithm (tsvdmi or tsvdmii)")
    parser.add_argument("-mtype",     "--mtype",     type=str,   help="Transformation matrix type (dct, eye, hosvd)")
    parser.add_argument("-k",         "--k",         type=int,   help="Slice rank for tsvdm-I")
    parser.add_argument("-tol",       "--tol",       type=float, help="Error tolerance for tsvdm-II")
    parser.add_argument("-dname",     "--dname",     type=str,   help="Data name (e.g. soccer, traffic, cfd)")
    parser.add_argument("-dfile",     "--dfile",     type=str,   help="Path to the data file or directory")
    parser.add_argument("-perm-mode", "--perm-mode", type=str,   help="Permutation of modes as a string of digits (e.g. '120'), so that the last mode becomes the transformation mode")
    args = parser.parse_args()

    alg       = args.alg
    mtype     = args.mtype
    k         = args.k
    tol       = args.tol
    dname     = args.dname
    dfile     = args.dfile
    # If no perm-mode is passed, default to identity permutation (no reordering of modes).
    # The actual tuple is deferred until the tensor is loaded, since we need to know the number of dimensions.
    perm_mode = tuple(int(c) for c in args.perm_mode) if args.perm_mode else None

    print("alg      :", alg)
    print("mtype    :", mtype)
    print("k        :", k)
    print("tol      :", tol)
    print("dname    :", dname)
    print("dfile    :", dfile)
    print("perm_mode:", perm_mode if perm_mode is not None else "default (identity)")

    if dname == "soccer":
        arr = read_soccer_data(dfile)
    elif dname == "traffic":
        arr = read_traffic_data(dfile)
    elif dname == "cfd":
        arr = read_cfd_data(dfile)
    else:
        raise ValueError(f"Unknown dname: {dname}")

    if perm_mode is None:
        perm_mode = tuple(range(arr.ndim))
    elif len(perm_mode) != arr.ndim:
        raise ValueError(f"perm_mode length {len(perm_mode)} does not match tensor dimensions {arr.ndim}")
    arr = np.transpose(arr, perm_mode)

    # Copy slice by slice into a new Fortran-order buffer to ensure column-major
    # memory layout is respected and the data is contiguous after the permutation.
    x = np.zeros(arr.shape, dtype=np.float64, order='F')
    for i in range(arr.shape[-1]):
        x[..., i] = arr[..., i]
    arr = x

    print("Tensor shape:", arr.shape)
    print("Tensor flags:\n", arr.flags)

    # Apply TTM on all modes except the first two.
    # For a 3-way tensor (soccer): ttm_modes = [2]
    # For a 4-way tensor (traffic): ttm_modes = [2, 3]
    # For a 5-way tensor (cfd): ttm_modes = [2, 3, 4]
    ttm_modes = list(range(2, arr.ndim))

    Ms  = []
    MTs = []
    t0 = time.perf_counter()
    if mtype == "dct":
        for mode in ttm_modes:
            n   = arr.shape[mode]
            DF  = np.asfortranarray(dct(np.eye(n, dtype=np.float64), axis=0, norm="ortho"))
            DFT = np.asfortranarray(DF.T)
            Ms.append(pystarm.Matrix(DF,  DF.shape[0],  DF.shape[1]))
            MTs.append(pystarm.Matrix(DFT, DFT.shape[0], DFT.shape[1]))
    elif mtype == "eye":
        ttm_modes = []
    elif mtype == "hosvd":
        # Run HOSVD once to get factor matrices for all modes at once
        ttb_tensor = ttb.tensor(arr)
        hosvd      = ttb.hosvd(ttb_tensor, tol=0)
        Us         = hosvd.factor_matrices
        # Extract the factor matrix for each TTM mode
        for mode in ttm_modes:
            U  = np.asfortranarray(Us[mode])
            UT = np.asfortranarray(Us[mode].T)
            Ms.append(pystarm.Matrix(U,  U.shape[0],  U.shape[1]))
            MTs.append(pystarm.Matrix(UT, UT.shape[0], UT.shape[1]))
    t1 = time.perf_counter()
    print("Time to generate transformation matrices:", t1 - t0)

    t0 = time.perf_counter()
    A = pystarm.Tensor(arr, len(arr.shape), arr.shape)
    t1 = time.perf_counter()
    print("Time to convert to pystarm tensor:", t1 - t0)

    if alg == "tsvdmi":
        (U_hat, S_hat, VT_hat) = tsvdm_I_compress(A, Ms, ttm_modes, k, True)
        print("Total buffer:", U_hat.getbuflen() + S_hat.getbuflen() + VT_hat.getbuflen())
        Atilde = tsvdm_I_reconstruct(U_hat, S_hat, VT_hat, MTs, ttm_modes, True)

    elif alg == "tsvdmii":
        (U_hat, S_hat, VT_hat) = tsvdm_II_compress(A, Ms, ttm_modes, tol, True)
        print("Total buffer:", U_hat.getbuflen() + S_hat.getbuflen() + VT_hat.getbuflen())
        Atilde = tsvdm_II_reconstruct(U_hat, S_hat, VT_hat, MTs, ttm_modes, True)

    arr_reconst = np.frombuffer(Atilde, dtype=np.float64).reshape(Atilde.getdims(), order='F', copy=False)

    arr_diff         = arr - arr_reconst
    norm_arr_diff    = np.linalg.norm(arr_diff)
    norm_arr         = np.linalg.norm(arr)
    norm_arr_reconst = np.linalg.norm(arr_reconst)

    print("Absolute err:", norm_arr_diff)
    print("Relative err:", norm_arr_diff / norm_arr)
    print("Norm of original tensor:", norm_arr)
    print("Norm of reconstructed tensor:", norm_arr_reconst)
