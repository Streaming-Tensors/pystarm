"""
ncep.py — Read NCEP Reanalysis pressure-level air temperature data into a
Fortran-order float64 tensor.

Settings are hard-coded for now (see SETTINGS section below).

NCEP air temperature files (air.{year}.nc) have four dimensions as stored by
xarray / NetCDF:

    (time, level, lat, lon)

    time  : 4 observations per day (6-hourly), so 1460 steps/year
            (1464 in leap years)
    level : 17 standard pressure levels
             [1000, 925, 850, 700, 600, 500, 400, 300, 250,
               200, 150, 100,  70,  50,  30,  20,  10] hPa
    lat   : 73 latitudes  (90°N → 90°S, 2.5° spacing)
    lon   : 144 longitudes (0° → 357.5°E, 2.5° spacing)

After reading we transpose to (lat, lon, level, time) so that spatial modes
are first and the concatenation (time) axis is last — consistent with how
other datasets are handled in experiments.py.

The final multi-year tensor is assembled by copying each year's data
slice-by-slice into a pre-allocated Fortran-order float64 buffer (same
pattern as read_cfd_data in experiments.py).
"""

import os
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# SETTINGS — hard-coded for now, will be generalised later
# ---------------------------------------------------------------------------

# Directory containing air.{year}.nc files (the pressure/ subdirectory
# created by dload-ncep-data.py)
DATA_DIR = "/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"

# Variable name inside the NetCDF files
VARIABLE = "air"

# Year range to load (inclusive)
YEAR_START = 1948
YEAR_END   = 1958

# ---------------------------------------------------------------------------


def read_ncep_air(data_dir: str, variable: str, year_start: int, year_end: int) -> np.ndarray:
    """
    Read NCEP Reanalysis pressure-level files for a range of years and return
    a single Fortran-order float64 tensor with shape (lat, lon, level, time).

    Each annual file is opened with xarray, the data variable is extracted as
    a NumPy array, converted to float64, and transposed from (time, level,
    lat, lon) → (lat, lon, level, time).  All years are then concatenated
    along the time axis and copied slice-by-slice into a contiguous
    Fortran-order buffer.

    Parameters
    ----------
    data_dir   : Directory containing files named {variable}.{year}.nc.
    variable   : NetCDF variable name (e.g. 'air').
    year_start : First year to load (inclusive).
    year_end   : Last year to load (inclusive).

    Returns
    -------
    x : np.ndarray, dtype float64, Fortran order, shape (lat, lon, level, time).
    """
    years = range(year_start, year_end + 1)
    per_year = []  # list of (lat, lon, level, time_for_this_year) arrays

    for year in years:
        filepath = os.path.join(data_dir, f"{variable}.{year}.nc")
        print(f"  Reading {os.path.basename(filepath)} ...", end=" ", flush=True)

        with xr.open_dataset(filepath) as ds:
            # Extract variable as numpy array: shape (time, level, lat, lon)
            raw = ds[variable].values

        # Convert to float64 (NCEP files are typically stored as int16 with a
        # scale_factor / add_offset that xarray already applies, but the
        # result may be float32 — ensure float64 explicitly)
        raw = raw.astype(np.float64)

        # Transpose (time, level, lat, lon) → (lat, lon, level, time)
        # so that spatial modes are first and the time axis (to concatenate
        # on) is last, matching the convention used in experiments.py.
        arr = np.transpose(raw, (2, 3, 1, 0))  # lat=2, lon=3, level=1, time=0

        print(f"shape={arr.shape}  dtype={arr.dtype}")
        per_year.append(arr)

    # Concatenate all years along the last (time) axis.
    # np.concatenate returns a C-order array; we re-copy below.
    full = np.concatenate(per_year, axis=-1)

    print(f"\nConcatenated shape (before Fortran copy): {full.shape}")

    # Copy slice-by-slice into a pre-allocated Fortran-order buffer.
    # This mirrors read_cfd_data: iterating over the last axis ensures each
    # 3-D slice is written in column-major order, producing a truly contiguous
    # Fortran-layout array in memory.
    x = np.zeros(full.shape, dtype=np.float64, order='F')
    for i in range(full.shape[-1]):
        x[..., i] = full[..., i]

    return x


if __name__ == "__main__":
    print(f"Loading NCEP '{VARIABLE}' data")
    print(f"  Directory  : {DATA_DIR}")
    print(f"  Years      : {YEAR_START}–{YEAR_END}")
    print()

    arr = read_ncep_air(DATA_DIR, VARIABLE, YEAR_START, YEAR_END)

    # --- Report ---
    ndim        = arr.ndim
    shape       = arr.shape
    bytes_total = arr.nbytes
    mb          = bytes_total / (1024 ** 2)
    gb          = bytes_total / (1024 ** 3)

    print()
    print("Tensor dimensions  :", ndim)
    print("Tensor shape       :", shape)
    print("  lat              :", shape[0])
    print("  lon              :", shape[1])
    print("  level            :", shape[2])
    print("  time (all years) :", shape[3])
    print(f"Memory             : {bytes_total:,} bytes  ({mb:.1f} MB  /  {gb:.3f} GB)")
    print("dtype              :", arr.dtype)
    print("Fortran-contiguous :", arr.flags['F_CONTIGUOUS'])
    print("C-contiguous       :", arr.flags['C_CONTIGUOUS'])
