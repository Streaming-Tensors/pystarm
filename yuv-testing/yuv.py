'''
This is a script used to measure the quality of compressions with .yuv files.

Note I load a collection of M-opt values, these were determined by a different program.
'''

import numpy as np
import pystarm
from tests.alg import tsvdm_I_compress, tsvdm_I_reconstruct


def load_yuv_as_tensor(filename: str, width: int, height: int, num_frames: int) -> np.ndarray:
    """
    Load a raw YUV 4:2:0 video file as a greyscale mode-3 tensor.
    The returned array is Fortran-contiguous (column-major) for compatibility
    with pystarm.
    """
    frame_size_y = width * height
    frame_size_uv = (width * height) // 2  # U and V combined
    total_frame_size = frame_size_y + frame_size_uv

    # Pre-allocate in Fortran order for pystarm compatibility
    T = np.zeros((height, width, num_frames), dtype=np.float64, order='F')

    with open(filename, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        max_frames = file_size // total_frame_size
        f.seek(0)

        if max_frames < num_frames:
            raise ValueError(
                f"File contains only {max_frames} frames, "
                f"but {num_frames} were requested."
            )

        for k in range(num_frames):
            f.seek(k * total_frame_size)
            y_data = f.read(frame_size_y)

            # Raw YUV stores pixels in raster-scan (row-major) order,
            # so reshape with C order to get correct spatial layout
            y_frame = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width), order='C')

            # Assignment into Fortran-order T handles the memory layout
            T[:, :, k] = y_frame.astype(np.float64)

    assert T.flags['F_CONTIGUOUS'], "Tensor must be Fortran-contiguous for pystarm"
    print(f"Loaded tensor of size {T.shape[0]} x {T.shape[1]} x {T.shape[2]}")
    return T

## TODO: Run and reconstruct for all matrices. 

if __name__ == "__main__":
    # Example usage with a standard CIF (352x288) test sequence
    # SOURCE OF VIDEO: https://media.xiph.org/video/derf/
    filename = "data/akiyo_cif.y4m"
    width = 352 # This is our n1
    height = 288 # n2
    num_frames = 200 # n3

    T = load_yuv_as_tensor(filename, width, height, num_frames)
    print(T[:3, :3, 0])
    A = pystarm.Tensor(T, T.ndim, T.shape) # A, mode, dims

    k_vals = np.arange(1,6)
    M_I = [pystarm.Matrix(np.eye(num_frames), num_frames, num_frames)]
    for k in k_vals:
        # Compress
        M_opt = np.genfromtxt(f"data/akiyo-M-opts/MATRIX-k-{k}.txt", delimiter=',')
        print(np.allclose(M_opt @ M_opt.T, np.eye(num_frames)))
        M_opt_pystarm = [pystarm.Matrix(M_opt, M_opt.shape[0], M_opt.shape[1])]
        U_hat, S_hat, VT_hat = tsvdm_I_compress(A, M_opt_pystarm, [2], k)

        # Reconstruct
        M_opt_inv = [pystarm.Matrix(M_opt.T, M_opt.shape[0], M_opt.shape[1])] # Using transpose, M is orthogonal, so they're the same
        A_tilde = tsvdm_I_reconstruct(U_hat, S_hat, VT_hat, M_opt_inv, [2], (width, height, num_frames))

        # Calculate error
        A_tilde_np = np.frombuffer(A_tilde, dtype=np.float64).reshape(A_tilde.getdims(), order='F', copy = False)
        diff = T - A_tilde_np
        diff_pystarm = pystarm.Tensor(diff, T.ndim, T.shape)
        print(diff_pystarm.norm() / A.norm())




    print(f"Tensor shape: {T.shape}")
    print(f"Tensor dtype: {T.dtype}")
    print(f"Value range:  [{T.min():.0f}, {T.max():.0f}]")
