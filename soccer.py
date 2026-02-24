
# import importlib.util, sys
# _spec = importlib.util.spec_from_file_location(
        # "pystarm",                      # the name Python uses — must match PYBIND11_MODULE name
        # "/global/homes/t/taufique/Codes/pystarm/pystarm_asan.cpython-311-x86_64-linux-gnu.so"      # the actual file to load
# )
# pystarm = importlib.util.module_from_spec(_spec)
# sys.modules["pystarm"] = pystarm    # registers it so "import pystarm" in alg.py also gets this
# _spec.loader.exec_module(pystarm)

import time
import numpy as np
import scipy as sp
from scipy.fft import dct
import cv2
import numpy as np
import matplotlib.pyplot as plt
from alg import tsvdm_I_compress
from alg import tsvdm_I_reconstruct
from alg import tsvdm_II_compress
from alg import tsvdm_II_reconstruct
import argparse
import pystarm
import pyttb as ttb

def video_to_gray_array(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # H x W
        frames.append(gray)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()

    t_h_w = np.stack(frames, axis=0)  # T x H x W
    # Convert to a tensor where the time mode is contiguous. 
    # In Fortran order which is the last dimension that needs to be contiguous
    # In this video data, we keep the height and width dimension to be in the same order, just push the time dimension to the end (1,2,0)
    # Because the numpy array that contains the video frames stacked one after another is having each frame as contiguous
    h_w_t = np.transpose(t_h_w, (1, 2, 0)) # H x W x T
    # print(h_w_t.flags)
    
    # Do not do reshape. It would cause x to not own it's data which causes potential memory error when the execution of the function ends
    x = np.zeros(h_w_t.shape, dtype=np.float64, order='F')
    for i in range(h_w_t.shape[2]):
        x[:,:,i] = h_w_t[:,:,i]
    # print(x.flags)

    # return np.asfortranarray(h_w_t)
    return x

def write_mp4_opencv(arr: np.ndarray, out_path: str, fps: float = 30.0,
                     codec: str = "mp4v") -> None:
    """
    arr: H x W x T (grayscale uint8) or H x W x 3 x T (RGB uint8).
    Writes out_path as mp4.
    """
    if arr.ndim == 3:
        H, W, T = arr.shape
        is_color = False
    elif arr.ndim == 4 and arr.shape[2] == 3:
        H, W, C, T = arr.shape
        is_color = True
    else:
        raise ValueError("arr must be HxWxT (grayscale) or HxWx3xT (RGB)")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H), isColor=is_color)

    for t in range(arr.shape[-1]):
        if not is_color:
            frame = arr[:, :, t]
            # ensure uint8 single-channel -> convert to BGR if writer expects color False it's fine
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            out.write(frame)  # OpenCV accepts single-channel for isColor=False
        else:
            frame = arr[:, :, :, t]  # H x W x 3 (RGB)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # convert RGB -> BGR for OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)
    out.release()


 # video_to_rgb_array(path, max_frames=None, to_float=False):
    # cap = cv2.VideoCapture(path)
    # frames = []
    # while True:
        # ret, frame = cap.read()
        # if not ret:
            # break
        # # frame is BGR; convert to RGB
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # H x W x 3
        # frames.append(rgb)
        # if max_frames and len(frames) >= max_frames:
            # break
    # cap.release()
    # arr = np.stack(frames, axis=0)  # T x H x W x 3
    # if to_float:
        # arr = arr.astype(np.float32) / 255.0
    # return np.asfortranarray(arr)   # convert to Fortran (column-major) order

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-alg", "--alg", type=str, help="Name of the algorithm")
    parser.add_argument("-mtype", "--mtype", type=str, help="Transformation matrix type")
    parser.add_argument("-k", "--k", type=int, help="Slice rank for tsvdm-I")
    parser.add_argument("-tol", "--tol", type=float, help="Error tolerance for tsvdm-II")
    args = parser.parse_args()
    
    alg = None
    mtype = None
    k = None
    tol = None

    alg = args.alg
    mtype = args.mtype
    if args.k is not None:
        k = args.k
    if args.tol is not None:
        tol = args.tol

    t0 = time.perf_counter()
    arr_gry = video_to_gray_array("data/iniesta.mp4")
    t1 = time.perf_counter()
    print("Time to read data into numpy:", t1-t0)
    arr_gry_shape = arr_gry.shape
    print("Tensor shape:", arr_gry.shape) 

    t0 = time.perf_counter()
    A = pystarm.Tensor(arr_gry, len(arr_gry_shape), arr_gry_shape)
    t1 = time.perf_counter()
    print("Time to convert to Pystarm tensor:", t1-t0)
    
    t0 = time.perf_counter()
    Ms = []
    MTs = []
    ttm_modes = [2] # Because soccer data is just a three way-tensor
    if mtype == "dct":
        for i in range(len(ttm_modes)):
            mode = ttm_modes[i]
            n = arr_gry_shape[mode]
            DC = dct(np.eye(n), axis=0, norm="ortho")
            DF = np.asfortranarray(DC)
            DFT = np.asfortranarray(DF.T)

            Ms.append(pystarm.Matrix(DF, DF.shape[0], DF.shape[1]))
            MTs.append(pystarm.Matrix(DFT, DFT.shape[0], DFT.shape[1]))
    elif mtype == "eye":
        ttm_modes = []
    elif mtype == "hosvd":
        ttb_tensor = ttb.tensor(arr_gry)
        hosvd = ttb.hosvd(ttb_tensor, tol=0)
        Us = hosvd.factor_matrices
        for i in range(len(ttm_modes)):
            mode = ttm_modes[i]
            U  = np.asfortranarray(Us[mode])
            UT = np.asfortranarray(Us[mode].T)
            Ms.append(pystarm.Matrix(U, U.shape[0], U.shape[1]))
            MTs.append(pystarm.Matrix(UT, UT.shape[0], UT.shape[1]))
    t1 = time.perf_counter()
    print("Time to generate transformation matrix:", t1-t0)
    
    Atilde = None
    filename = None
    if alg == "tsvdmi":
        (U_hat,S_hat,VT_hat) = tsvdm_I_compress(A, Ms, ttm_modes, k, True)
        print("Total buffer:", U_hat.getbuflen() + S_hat.getbuflen() + VT_hat.getbuflen() )
        Atilde = tsvdm_I_reconstruct(U_hat, S_hat, VT_hat, MTs, ttm_modes, True)
        filename = "data/iniesta_reconst" + "-" + alg + "-" + mtype + "-" + str(k) + ".mp4"

    elif alg == "tsvdmii":
        (U_hat,S_hat,VT_hat) = tsvdm_II_compress(A, Ms, ttm_modes, tol, True)
        print("Total buffer:", U_hat.getbuflen() + S_hat.getbuflen() + VT_hat.getbuflen() )
        Atilde = tsvdm_II_reconstruct(U_hat, S_hat, VT_hat, MTs,ttm_modes, True)
        filename = "data/iniesta_reconst" + "-" + alg + "-" + mtype + "-" + str(args.tol) + ".mp4"

    arr_reconst = np.frombuffer(Atilde, dtype=np.float64).reshape(Atilde.getdims(), order='F', copy = False)
    # write_mp4_opencv(arr_reconst, filename)

    arr_diff = arr_gry - arr_reconst
    norm_arr_diff = np.linalg.norm(arr_diff)
    norm_arr_gry = np.linalg.norm(arr_gry)
    norm_arr_reconst = np.linalg.norm(arr_reconst)

    print("Absolute err:", norm_arr_diff)
    print("Relative err:", norm_arr_diff/norm_arr_gry)
    print("Norm of original array:", norm_arr_gry)
    print("Norm of reconstructed array:", norm_arr_reconst)
