import argparse
import os
import time
import numpy as np
import h5py
import cv2
import xarray as xr
from scipy.fft import dct
from sklearn.utils.extmath import randomized_svd
import pystarm
import pyttb as ttb
from pystarm.alg import tsvdm_I_compress, tsvdm_I_reconstruct
from pystarm.alg import tsvdm_II_compress, tsvdm_II_reconstruct


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
    arr = data.reshape((height, width, 3, num_frames), order='F')
    return arr / np.linalg.norm(arr)


def read_traffic_gray_data(filepath):
    """Read a traffic binary file (.bin), convert to grayscale, and return an H x W x T tensor.

    Grayscale conversion uses BT.601 luminance weights matching MATLAB's im2gray:
      I = 0.298936021293775 * R + 0.587043074451121 * G + 0.114020904255103 * B
    Pixel values in traffic.bin are float64 in [0, 255] (cast from uint8 by traffic_writer.m).
    The result is normalized by its Frobenius norm.
    """
    with open(filepath, 'rb') as f:
        height = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        width  = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        _fps   = np.frombuffer(f.read(8), dtype=np.float64)[0]
        data   = np.frombuffer(f.read(), dtype=np.float64)

    frame_size = height * width * 3
    num_frames = len(data) // frame_size
    color = data.reshape((height, width, 3, num_frames), order='F')

    # BT.601 luminance weights — matches MATLAB im2gray for uint8 input
    gray = (0.298936021293775 * color[:, :, 0, :] +
            0.587043074451121 * color[:, :, 1, :] +
            0.114020904255103 * color[:, :, 2, :])
    return gray / np.linalg.norm(gray)


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


def read_dcmall_data(filepath):
    import tifffile
    arr = tifffile.imread(filepath).astype(np.float64)
    print("DC Mall raw shape:", arr.shape)
    return arr


def read_xray_data(filepath):
    """Read xray crystallography data from a .npy file and return a Fortran-order float64 tensor."""
    arr = np.load(filepath).astype(np.float64)
    print("Xray raw shape:", arr.shape)
    x = np.zeros(arr.shape, dtype=np.float64, order='F')
    x[...] = arr
    return x


def read_ncep_air(data_dir, variable, year_start, year_end):
    """Read NCEP Reanalysis pressure-level files for a range of years and return
    a single Fortran-order float64 tensor with shape (lat, lon, level, time).

    Each annual file is opened with xarray, the data variable is extracted as
    a NumPy array, converted to float64, and transposed from (time, level,
    lat, lon) -> (lat, lon, level, time).  All years are then concatenated
    along the time axis and copied slice-by-slice into a contiguous
    Fortran-order buffer.
    """
    years = range(year_start, year_end + 1)
    per_year = []

    for year in years:
        filepath = os.path.join(data_dir, f"{variable}.{year}.nc")
        print(f"  Reading {os.path.basename(filepath)} ...", end=" ", flush=True)

        with xr.open_dataset(filepath) as ds:
            raw = ds[variable].values

        raw = raw.astype(np.float64)
        # Transpose (time, level, lat, lon) -> (lat, lon, level, time)
        arr = np.transpose(raw, (2, 3, 1, 0))
        print(f"shape={arr.shape}  dtype={arr.dtype}")
        per_year.append(arr)

    full = np.concatenate(per_year, axis=-1)
    print(f"\nConcatenated shape (before Fortran copy): {full.shape}")

    x = np.zeros(full.shape, dtype=np.float64, order='F')
    for i in range(full.shape[-1]):
        x[..., i] = full[..., i]
    return x


def read_ncep_air_6(data_dir, variable, year_start, year_end):
    """Read NCEP Reanalysis pressure-level files for a range of years and return
    a 6-way Fortran-order float64 tensor with shape (lat, lon, level, tod, doy, year).

    tod  = time-of-day index (4 observations per day, 6-hourly)
    doy  = day-of-year index (365, Feb 29 dropped from leap years)
    year = year index

    Each annual file is transposed from (time, level, lat, lon) -> (lat, lon, level, time),
    Feb 29 (time steps 236-239) is dropped from leap years, and the time axis of length
    1460 is reshaped into (tod=4, doy=365).  All years are then stacked along a new
    trailing year axis.
    """
    import calendar

    STEPS_PER_DAY = 4
    DAYS_PER_YEAR = 365
    STEPS_PER_YEAR = STEPS_PER_DAY * DAYS_PER_YEAR  # 1460

    # Feb 29 starts at step 59*4 = 236 (0-indexed) in a leap year
    FEB29_START = 59 * STEPS_PER_DAY
    FEB29_END   = FEB29_START + STEPS_PER_DAY  # exclusive

    years = range(year_start, year_end + 1)
    per_year = []

    for year in years:
        filepath = os.path.join(data_dir, f"{variable}.{year}.nc")
        print(f"  Reading {os.path.basename(filepath)} ...", end=" ", flush=True)

        with xr.open_dataset(filepath) as ds:
            raw = ds[variable].values

        raw = raw.astype(np.float64)
        # Transpose (time, level, lat, lon) -> (lat, lon, level, time)
        arr = np.transpose(raw, (2, 3, 1, 0))

        # Drop Feb 29 from leap years
        if calendar.isleap(year):
            arr = np.concatenate([arr[..., :FEB29_START], arr[..., FEB29_END:]], axis=-1)

        if arr.shape[-1] != STEPS_PER_YEAR:
            raise ValueError(f"Year {year}: expected {STEPS_PER_YEAR} time steps after dropping Feb 29, got {arr.shape[-1]}")

        # Reshape time axis (1460,) -> (tod=4, doy=365)
        lat, lon, level, _ = arr.shape
        arr = arr.reshape((lat, lon, level, STEPS_PER_DAY, DAYS_PER_YEAR), order='F')

        print(f"shape={arr.shape}  dtype={arr.dtype}")
        per_year.append(arr)

    # Stack along a new trailing year axis -> (lat, lon, level, tod, doy, nyears)
    full = np.stack(per_year, axis=-1)
    print(f"\nStacked shape (before Fortran copy): {full.shape}")

    x = np.zeros(full.shape, dtype=np.float64, order='F')
    for i in range(full.shape[-1]):
        x[..., i] = full[..., i]
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-alg",       "--alg",       type=str,   help="Name of the algorithm (tsvdmi or tsvdmii)")
    parser.add_argument("-mtype",     "--mtype",     type=str,   help="Transformation matrix type (dct, eye, hosvd)")
    parser.add_argument("-k",         "--k",         type=int,   help="Slice rank for tsvdm-I")
    parser.add_argument("-tol",       "--tol",       type=float, help="Error tolerance for tsvdm-II or EOF")
    parser.add_argument("-k-max",     "--k-max",     type=int,   default=1000, help="Max rank for randomized SVD (EOF only)")
    parser.add_argument("-dname",     "--dname",     type=str,   help="Data name (e.g. soccer, traffic, cfd, xray)")
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

    k_max     = args.k_max

    print("alg      :", alg)
    print("mtype    :", mtype)
    print("k        :", k)
    print("tol      :", tol)
    print("dname    :", dname)
    print("dfile    :", dfile)
    print("perm_mode:", perm_mode if perm_mode is not None else "default (identity)")

    if dname == "soccer":
        arr = read_soccer_data(dfile)
    elif dname == "traffic-color":
        arr = read_traffic_data(dfile)
    elif dname == "traffic-gray":
        arr = read_traffic_gray_data(dfile)
    elif dname == "cfd":
        arr = read_cfd_data(dfile)
    elif dname == "dcmall":
        arr = read_dcmall_data(dfile)
    elif dname == "ncep-air":
        arr = read_ncep_air(dfile, "air", 1948, 1957)
    elif dname == "ncep-air-6":
        arr = read_ncep_air_6(dfile, "air", 1948, 1957)
    elif dname == "xray":
        arr = read_xray_data(dfile)
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

    original_size = 1
    for d in arr.shape:
        original_size *= d

    if alg == "tsvdmi":
        (U_hat, S_hat, VT_hat) = tsvdm_I_compress(A, Ms, ttm_modes, k, True)
        print("Compression ratio:", original_size / (U_hat.getbuflen() + S_hat.getbuflen() + VT_hat.getbuflen()))
        Atilde = tsvdm_I_reconstruct(U_hat, S_hat, VT_hat, MTs, ttm_modes, arr.shape, True)

    elif alg == "tsvdmii":
        (U_hat, S_hat, VT_hat) = tsvdm_II_compress(A, Ms, ttm_modes, tol, True)
        print("Compression ratio:", original_size / (U_hat.getbuflen() + S_hat.getbuflen() + VT_hat.getbuflen()))
        Atilde = tsvdm_II_reconstruct(U_hat, S_hat, VT_hat, MTs, ttm_modes, arr.shape, True)

    elif alg == "eof":
        # Unfold tensor to 2D: (all spatial modes flattened, time)
        orig_shape = arr.shape
        ntime  = orig_shape[-1]
        nspace = original_size // ntime
        A2d = arr.reshape((nspace, ntime), order='F')

        print(f"\nRunning randomized SVD (k_max={k_max})...")
        t0 = time.perf_counter()
        U, s, VT = randomized_svd(A2d, n_components=k_max, random_state=0)
        print(f"SVD time: {time.perf_counter() - t0:.1f}s")

        energy_total      = np.sum(s ** 2)
        energy_cumulative = np.cumsum(s ** 2)
        energy_threshold  = (1.0 - tol ** 2) * energy_total
        k_star = int(np.searchsorted(energy_cumulative, energy_threshold) + 1)
        k_star = min(k_star, k_max)
        print(f"k* (effective rank): {k_star}  (out of k_max={k_max})")

        U  = U[:,  :k_star]
        s  = s[    :k_star]
        VT = VT[:k_star, :]

        compressed_size = U.size + s.size + VT.size
        print("Compression ratio:", original_size / compressed_size)

        A2d_reconst = (U * s) @ VT
        arr_reconst = A2d_reconst.reshape(orig_shape, order='F')

        arr_diff         = arr - arr_reconst
        norm_arr_diff    = np.linalg.norm(arr_diff)
        norm_arr         = np.linalg.norm(arr)
        norm_arr_reconst = np.linalg.norm(arr_reconst)

        print("Absolute err:", norm_arr_diff)
        print("Relative err:", norm_arr_diff / norm_arr)
        print("Norm of original tensor:", norm_arr)
        print("Norm of reconstructed tensor:", norm_arr_reconst)

    if alg in ("tsvdmi", "tsvdmii"):
        arr_reconst = np.frombuffer(Atilde, dtype=np.float64).reshape(Atilde.getdims(), order='F', copy=False)

        arr_diff         = arr - arr_reconst
        norm_arr_diff    = np.linalg.norm(arr_diff)
        norm_arr         = np.linalg.norm(arr)
        norm_arr_reconst = np.linalg.norm(arr_reconst)

        print("Absolute err:", norm_arr_diff)
        print("Relative err:", norm_arr_diff / norm_arr)
        print("Norm of original tensor:", norm_arr)
        print("Norm of reconstructed tensor:", norm_arr_reconst)
