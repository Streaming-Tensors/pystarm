import time
import pystarm
import numpy as np
import scipy as sp
from scipy.fft import dct
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    print(h_w_t.flags)
    
    # Do not do reshape. It would cause x to not own it's data which causes potential memory error
    x = np.zeros(h_w_t.shape, dtype=np.float64, order='F')
    for i in range(h_w_t.shape[2]):
        x[:,:,i] = h_w_t[:,:,i]

    print(x.flags)

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

def tsvdm_3way(A, M, Minv, k):
    t0 = time.perf_counter()
    A_hat = pystarm.ttm(A, M, len(A.getdims())-1) 
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for TTM A x M:", t1-t0)
    
    # A_hat_np = np.frombuffer(A_hat, dtype=np.float64).reshape(A_hat.getdims(), order='F', copy = False)
    # plt.imshow(A_hat_np[:,:,500], cmap='viridis')
    # plt.savefig('data/iniesta_500_A_hat.png')
    # A2 = pystarm.ttm(A_hat, Minv, len(A_hat.getdims())-1)
    # A2_np = np.frombuffer(A2, dtype=np.float64).reshape(A2.getdims(), order='F', copy = False)
    # plt.imshow(A2_np[:,:,500], cmap='viridis')
    # plt.savefig('data/iniesta_500_A2.png')

    t0 = time.perf_counter()
    U_hat, S_hat, V_hat = pystarm.slicewise_svdx(A_hat, k)
    # U_hat, S_hat, V_hat = pystarm.slicewise_svd(A_hat)
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for slicewise SVD:", t1-t0)

    t0 = time.perf_counter()
    U = pystarm.ttm(U_hat, Minv, len(U_hat.getdims())-1 )
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for TTM U_hat x Minv:", t1-t0)

    t0 = time.perf_counter()
    S = pystarm.matmul(S_hat, Minv )
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for matmul S_hat x Minv:", t1-t0)

    t0 = time.perf_counter()
    V = pystarm.ttm(V_hat, Minv, len(U_hat.getdims())-1 )
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for TTM V_hat x Minv:", t1-t0)
    
    return (U,S,V)
    


if __name__ == "__main__":
    # usage
    t0 = time.perf_counter()
    arr_gry = video_to_gray_array("data/iniesta.mp4")
    # rng = np.random.default_rng(seed=42)
    # arr_gry = rng.random(arr_gry.shape, dtype=np.float64)
    t1 = time.perf_counter()
    print("Time to read data into numpy:", t1-t0)
    arr_gry_shape = arr_gry.shape
    # arr_gry_shape = (arr_gry_shape[0], arr_gry_shape[1], 10) 
    print("Grayscale tensor shape:", arr_gry.shape) 

    # print(arr_gry[:,:,1])
    # plt.imshow(arr_gry[:,:,500], cmap='viridis')
    # plt.savefig('data/iniesta_500.png')

    t0 = time.perf_counter()
    A = pystarm.Tensor(arr_gry, len(arr_gry_shape), arr_gry_shape)
    # A = pystarm.Tensor(arr_gry, len(arr_gry_shape), (arr_gry_shape[0], arr_gry_shape[1], 10))
    t1 = time.perf_counter()
    print("Time to convert to Pystarm tensor:", t1-t0)

    t0 = time.perf_counter()
    n = arr_gry_shape[-1]
    mat_nelm = n * n
    mat_dims = (n, n)
    mat_ndim = len(mat_dims)
    # arr1 = np.arange(n*n, dtype=np.float64).reshape(mat_dims, order='F')
    DC = dct(np.eye(n), axis=0, norm="ortho")
    DF = np.asfortranarray(DC)
    DFT = np.asfortranarray(DF.T)

    M = pystarm.Matrix(DF, mat_dims[0], mat_dims[1])
    MT = pystarm.Matrix(DFT, mat_dims[0], mat_dims[1])
    # I = np.asfortranarray(np.identity(n))
    # M = pystarm.Matrix(I, n, n)
    # MT = pystarm.Matrix(I, n, n)
    t1 = time.perf_counter()
    print("Time to generate the DCT matrix:", t1-t0)

    (U,S,VT) = tsvdm_3way(A, M, MT, 100)
    print(U.getdims())
    print(S.getdims())
    print(VT.getdims())

    # U_np = np.frombuffer(U, dtype=np.float64).reshape(U.getdims(), order='F', copy = False)
    # S_np = np.frombuffer(S, dtype=np.float64).reshape(S.getdims(), order='F', copy = False)
    # VT_np = np.frombuffer(VT, dtype=np.float64).reshape(VT.getdims(), order='F', copy = False)
    # print(U_np.shape)
    # print(S_np.shape)
    # print(VT_np.shape)

    t0 = time.perf_counter()
    Atilde = pystarm.slicewise_matmul(U, S, VT)
    t1 = time.perf_counter()
    print("Time to reconstruct:", t1-t0)

    arr_reconst = np.frombuffer(Atilde, dtype=np.float64).reshape(Atilde.getdims(), order='F', copy = False)
    # # write_mp4_opencv(arr_reconst, "data/iniesta_reconst.mp4")

    err = np.linalg.norm(np.abs(arr_reconst - arr_gry))
    norm_arr = np.linalg.norm(arr_gry)
    norm_rec = np.linalg.norm(arr_reconst)

    print(err, norm_arr, norm_rec)
