# pystarm — Project Guide for Claude

## What is pystarm

pystarm is a Python/C++ library that wraps matrix and tensor operations via pybind11 and Intel MKL. It is used to implement and benchmark t-SVDM (Tensor SVD with Matrix Transform) algorithms for tensor compression.

## Reference Papers

This work closely follows these two papers:

- **PNAS publication**: "Tensor-tensor algebra for optimal representation and compression of multiway data" — Misha E. Kilmer, Lior Horesh, Haim Avron, and Elizabeth Newman. https://www.pnas.org/doi/epub/10.1073/pnas.2015851118
- **ArXiv tech report**: "Tensor-Tensor Products for Optimal Representation and Compression" — Misha Kilmer, Lior Horesh, Haim Avron, and Elizabeth Newman. https://arxiv.org/abs/2001.00046

## Paper Summary

### Core Idea

The central question of the papers is: is it better to compress data as a matrix (2D) or as a tensor (multi-dimensional array)? The papers prove that treating data as a tensor and compressing via t-SVDM is provably at least as good as, and often strictly better than, matrix SVD.

### The t-SVDM Framework

Any invertible matrix **M** defines a tensor-tensor product (called ?M). The compression proceeds as:
1. Apply M along the transformation modes of the tensor (go to transform domain)
2. Perform independent matrix SVDs on each frontal slice in that domain
3. Apply M⁻¹ to reconstruct (inverse transform)

The original t-product (Kilmer & Martin, 2011) is the special case where M is the DFT matrix. The papers generalize this to any invertible M, with the key requirement that M must be a **non-zero multiple of a unitary (orthogonal) matrix** for the Eckart-Young optimality theorem to hold.

### Choices of M

| M | Notes |
|---|---|
| DFT | Original t-product, complex arithmetic |
| DCT | Real-valued, good for spatially/temporally smooth data |
| Wavelet | Orthogonal wavelet matrix |
| HOSVD factor Z | M = Z^T recovers HOSVD as a special case |
| Random orthogonal | No benefit — used as a negative baseline in experiments |
| Identity (eye) | No inter-slice mixing, weakest compression |

A good M concentrates the energy of the tensor into fewer, larger singular values in the transform domain (faster decay), enabling better compression at the same error level.

### Energy

The **energy** of a tensor is its squared Frobenius norm — the sum of squares of all its entries. By Corollary 6 of the paper, this equals the sum of squared singular values across all slices in the transform domain. Singular values therefore directly partition the total energy, and retaining the largest ones retains the most energy.

### Algorithms

**t-SVDM / tsvdmi** (Algorithm 2 in paper): Fix a global rank **k**. Transform, truncate every slice to rank k, inverse transform. Eckart-Young optimal: best rank-k approximation in Frobenius norm. Weakness: treats all slices equally — may over-allocate for easy slices and under-allocate for hard ones.

**t-SVDMII / tsvdmii** (Algorithm 3 in paper): Fix an energy level **γ** (e.g. 0.99 = retain 99% of energy). Compute all singular values across all slices globally, sort them, find a threshold τ such that retained values cover energy γ. Each slice keeps a different number of singular values (multi-rank ρ). Strictly better compression than t-SVDM: same or smaller error, smaller or equal storage (Theorem 17).

### Key Theoretical Results

- **Eckart-Young for t-SVDM** (Theorem 10): rank-k t-SVDM is the best rank-k approximation under ?M — analogous to truncated matrix SVD.
- **t-SVDMII is also optimal** (Theorem 11): multi-rank ρ approximation is best possible at that multi-rank.
- **Tensors beat matrices** (Theorem 15): t-SVDM error ≤ matrix SVD error for the same truncation parameter k. Strict inequality is possible.
- **t-rank ≤ matrix rank** (Theorem 13): A good M can reveal t-rank << matrix rank.
- **HOSVD is a special case** (Theorem 18 & 19): HOSVD is a specific instance of ?M, and truncated HOSVD is provably suboptimal compared to truncated t-SVDM.

### Extension to Higher-Order Tensors

For 4-way and higher tensors, different transforms M, B are applied along different modes, and slicewise SVDs are performed in the multi-transformed domain (Algorithm 5). This is what `experiments.py` implements — `ttm_modes = list(range(2, arr.ndim))` applies transforms on all modes beyond the first two.

### Numerical Results

Tested on Yale face data, traffic video (120×160×120), and hyperspectral images. In all cases t-SVDMII with DCT or wavelet outperforms matrix SVD and HOSVD at the same compression ratio. The traffic video experiment is directly relevant to this codebase.

## Algorithms

Two variants of t-SVDM are implemented:

- **tsvdm-I** (`tsvdmi`): Fixed-rank compression. Takes a rank parameter `k`.
- **tsvdm-II** (`tsvdmii`): Tolerance-based compression. Takes an error tolerance `tol`.

Both variants follow the same structure: apply a transformation (TTM) on all modes except the first two, perform a slicewise SVD, then reconstruct via inverse TTM.

## Key Conventions

- All data is **64-bit float (float64)**.
- All arrays and tensors are in **Fortran (column-major) order**.
- pystarm objects (`Tensor`, `Matrix`) must be **explicitly freed** by calling `.clear()` — ownership is on the C++ side.
- `np.frombuffer(pystarm_obj, dtype=np.float64)` gives a read-only 1D view of the internal buffer. Use `.getdims()` to get the shape and reshape with `order='F'`.
- `slicewise_svd` / `slicewise_svdx` treat `dims[0]` and `dims[1]` as matrix dimensions and `dims[2:]` as slice dimensions. They always return a **3D tensor**, collapsing all modes ≥ 2 into a single `nslices` dimension.

# MKL + OpenMP Thread Settings
 
 ```bash
 export OMP_NUM_THREADS=64
 export MKL_NUM_THREADS=64
 export MKL_DYNAMIC=FALSE
 ```
  
`cblas_dgemm` is called from a serial region and uses all 64 cores via `MKL_NUM_THREADS=64`.

`dgesvd` is called inside a `#pragma omp parallel for` with `mkl_set_num_threads_local(1)`
bracketing each call, so 64 OpenMP threads run in parallel each executing a sequential SVD.
This overrides `MKL_NUM_THREADS=64` per-thread without affecting the global setting used by `cblas_dgemm`.

## Directory Structure

```
cpp/                    C++ source code
  matrix.hpp/cpp        Matrix class
  tensor.hpp/cpp        Tensor class
  ops.cpp               Matrix and tensor operations (TTM, slicewise SVD, etc.)
  starm.cpp             pybind11 Python bindings
alg.py                  tsvdm-I and tsvdm-II compress and reconstruct functions
experiments.py          Unified experiment script for all datasets (tsvdmi, tsvdmii, eof)
ncep_ranks.py           Run tsvdmii on NCEP data and analyze per-slice rank distributions
scripts/ncep/ncep-air_tsvdmii.py     Load NCEP air data, compress+reconstruct with tsvdmii, save reconstruction
scripts/ncep/ncep-air_eof.py         Load NCEP air data, compress+reconstruct with EOF (randomized SVD), save reconstruction
scripts/ncep/ncep-air_extremes.py    Compare EOF vs tsvdmii reconstruction quality at extreme temperature events
ncep-air_snapshot.py    Plot air temperature reconstruction error and values at a single time snapshot
ncep-slp_tsvdmii.py     Load NCEP SLP data, compress+reconstruct with tsvdmii, save reconstruction
ncep-slp_eof.py         Load NCEP SLP data, compress+reconstruct with EOF (randomized SVD), save reconstruction
ncep-slp_extremes.py    Compare EOF vs tsvdmii reconstruction quality at extreme SLP events
ncep-slp_snapshot.py    Plot SLP reconstruction error and values at a single time snapshot
scripts/
  experiments.sh        Batch-run experiments (Python and/or MATLAB); set RUN_PYTHON/RUN_MATLAB switches at top
  parse_logs.py         Parses experiment log files into experiments.csv
  hosvd_experiment.m    MATLAB HOSVD experiment script (single run, called by experiments.sh)
  plot_alg-compare.py   Plot relative error vs compression ratio for a given dataset
  plot_ncep-air_compression.py      Compression ratio vs relative error for ncep-air (tsvdmi-dct, tsvdmii-dct, tsvdmii-eye, EOF)
  plot_ncep-air_compression_4vs6.py Compression ratio vs relative error: ncep-air (4-way) vs ncep-air-6 (6-way), tsvdmii-dct
  plot_ncep-air_ttm_scaling.py      Compress TTM strong scaling for ncep-air (mode 2 and mode 3), tsvdmii-dct tol=0.01
  plot_ncep-air_ttm_scaling_4vs6.py Compress TTM strong scaling: 4-way vs 6-way stacked bars per mode
  plot_ncep-air_ranks.py            Per-slice rank barplot: dct (top) vs eye (bottom), tsvdmii tol=0.01
  plot_ncep_slp_compression.py      Plot EOF vs tsvdmii compression curve for ncep-slp
  asan-run-prep.sh      Sets up environment for running with AddressSanitizer
  dload-ncep-data.py    Multi-threaded download script for NCEP Reanalysis NetCDF files
  ncep.py               Standalone NCEP data reader (settings hard-coded at top)
known_issues/           Known bugs and debugging notes
test.py                 Unit tests and usage examples for pystarm
```

## Datasets

| dname          | Format        | Shape            | Normalized | Reader function          | Perm modes    |
|----------------|---------------|------------------|------------|--------------------------|---------------|
| soccer         | .mp4 (video)  | H × W × T                  | No  | `read_soccer_data`       | 012           |
| traffic-color  | .bin (binary) | H × W × 3 × T              | Yes | `read_traffic_data`      | 0123, 0321    |
| traffic-gray   | .bin (binary) | H × W × T                  | Yes | `read_traffic_gray_data` | 012, 021, 120 |
| dcmall         | .tif (image)  | 191 × 1280 × 307            | No  | `read_dcmall_data`       | 021           |
| cfd            | .h5 (HDF5)    | 5-way tensor                | No  | `read_cfd_data`          | 01234         |
| ncep-air       | .nc (NetCDF)  | 73 × 144 × 17 × T (per yr) | No  | `read_ncep_air`           | —             |
| ncep-slp       | .nc (NetCDF)  | 73 × 144 × T (per yr)      | No  | `read_ncep_slp`           | —             |
| xray           | .npy (numpy)  | 300 × 400 × 400             | No  | `read_xray_data`          | 012           |

- `traffic-gray` is derived from `traffic.bin` using BT.601 luminance weights matching MATLAB's `im2gray`.
- `dcmall` is the DC Mall hyperspectral image from `$CFS/m4293/HSI/Hyperspectral_Project/dc.tif`. It is not normalized.
- Normalized datasets (traffic-color, traffic-gray) are divided by their Frobenius norm before compression.

### traffic.bin binary format
Header: `uint32 height`, `uint32 width`, `float64 fps`
Body: float64 values in MATLAB column-major order, shape `(H, W, 3, T)`

### NCEP air temperature dimensions
Each annual NetCDF file (`air.{year}.nc`) has the following axes as read by xarray:

| Axis (raw) | Name  | Size | Description |
|---|---|---|---|
| 0 | time  | 1460 or 1464 | 6-hourly observations (4/day; 1464 in leap years) |
| 1 | level | 17           | Pressure levels (hPa): 1000 925 850 700 600 500 400 300 250 200 150 100 70 50 30 20 10 |
| 2 | lat   | 73           | 90°N → 90°S, 2.5° spacing |
| 3 | lon   | 144          | 0° → 357.5°E, 2.5° spacing |

`read_ncep_air` transposes to **(lat, lon, level, time)** so spatial modes come first,
matching the convention in `experiments.py`.  For 10 years (1948–1957) the tensor
shape is **73 × 144 × 17 × 14612** (~19.46 GB as float64).

The function to call is `read_ncep_air(data_dir, variable, year_start, year_end)` defined in `experiments.py`.

### NCEP SLP dimensions
Each annual NetCDF file (`slp.{year}.nc`) has the following axes as read by xarray:

| Axis (raw) | Name  | Size | Description |
|---|---|---|---|
| 0 | time  | 1460 or 1464 | 6-hourly observations (4/day; 1464 in leap years) |
| 1 | lat   | 73           | 90°N → 90°S, 2.5° spacing |
| 2 | lon   | 144          | 0° → 357.5°E, 2.5° spacing |

`read_ncep_slp` transposes to **(lat, lon, time)**. For 31 years (1985–2015) the tensor
shape is roughly **73 × 144 × 181,056** (~1.6 GB as float64).

Data location: `/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/surface`

## experiments.py

Unified script for running experiments on any dataset.

```bash
python experiments.py -alg <tsvdmi|tsvdmii|eof> -mtype <dct|eye|hosvd> \
    -k <rank> -tol <tolerance> -k-max <max_svd_rank> \
    -dname <soccer|traffic-color|traffic-gray|dcmall|cfd|ncep-air|ncep-slp> -dfile <path> \
    -perm-mode <digit string e.g. 0123>
```

- `-perm-mode`: digit string passed to `np.transpose` to reorder modes so transformation modes are last. e.g. `"0123"` → `(0,1,2,3)`.
- `ttm_modes = list(range(2, arr.ndim))` — transformation is applied on all modes except the first two.
- After permutation, a Fortran-order copy is made slice-by-slice along the last axis.
- For `mtype=eye`, no TTM is applied (identity transform is a no-op).
- For `mtype=hosvd`, HOSVD is run once via pyttb to get all factor matrices, then applied per mode.
- For `alg=eof`, `-mtype` and `-perm-mode` are ignored. The tensor is unfolded to `(nspace, ntime)` and randomized SVD is applied. `-k-max` sets the max SVD rank (default 1000). Log filename uses `none_none` for the mtype/perm_mode fields.

## Running Experiments

All commands are run from the code root directory.

Edit the switches at the top of `scripts/experiments.sh` before running:
- `RUN_PYTHON` / `RUN_MATLAB` — toggle which experiments to run
- `OUTPUT_DIR` — directory where log files are written (default: `$SCRATCH/pystarm`)
- `TENSOR_TOOLBOX_PATH` — path to MATLAB Tensor Toolbox (required for MATLAB runs)
- `DNAME` loop and `K_VALUES` / `PERM_MODES` — select datasets and parameter ranges

```bash
bash scripts/experiments.sh
```

Each run produces one log file per parameter combination named:
```
{dname}_{alg}_{k_or_tol}_{mtype}_{perm_mode}_{omp_threads}
```

## Parsing Logs

Parses all log files in a directory into a single CSV:

```bash
python scripts/parse_logs.py <logdir> <outfile.csv>
```

Example:
```bash
python scripts/parse_logs.py $SCRATCH/pystarm/logs experiments.csv
```

The CSV is written to the code root and read by the plotting script.

## Plotting

Plot relative error vs compression ratio for a given dataset:

```bash
python scripts/plot_alg-compare.py --dname <dname>
```

Example:
```bash
python scripts/plot_alg-compare.py --dname dcmall
```

Output is saved to `plots/{dname}_alg-compare.pdf`. The script reads `experiments.csv` from the current directory and skips any algorithm series that has no data.

## NCEP Reanalysis Data

NCEP Reanalysis data is a long-running atmospheric reanalysis product from NOAA, covering
1948–present. Files are served over FTP from NOAA PSL:

```
ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis/{level}/{variable}.{year}.nc
```

Each `.nc` file is one year of a single variable at a given level. Files are in NetCDF format.

### Level directories

| Level | Contents |
|---|---|
| `pressure` | Pressure-level data (17 standard levels: 1000–10 hPa) |
| `surface` | Surface or near-surface variables |
| `surface_gauss` | Surface data on Gaussian grid |
| `tropopause` | Tropopause-level data |
| `other_gauss` | Other Gaussian-grid variables |

### Common variables

| Variable | Level | Description |
|---|---|---|
| `air` | pressure | Air temperature |
| `hgt` | pressure | Geopotential height |
| `uwnd` | pressure | Zonal (U) wind |
| `vwnd` | pressure | Meridional (V) wind |
| `omega` | pressure | Vertical velocity (Pa/s) |
| `rhum` | pressure | Relative humidity |
| `shum` | pressure | Specific humidity |
| `slp` | surface | Sea level pressure |
| `air.sig995` | surface | Near-surface air temperature |
| `pres.sfc` | surface | Surface pressure |
| `pr_wtr.eatm` | surface | Precipitable water |

### Downloading: `scripts/dload-ncep-data.py`

Multi-threaded download script. Spawns one thread per year (up to the CPU core count) so
all years of a variable are fetched in parallel.

```bash
python scripts/dload-ncep-data.py \
    --year-start <YEAR> --year-end <YEAR> \
    --level <level> --variable <variable> \
    --outdir <path>
```

All parameters are required; `--help` prints full usage.

The script automatically creates a subdirectory named after `--level` inside `--outdir`,
mirroring the FTP layout. Example: `--outdir ./ncep --level pressure` saves files to
`./ncep/pressure/{variable}.{year}.nc`. This keeps data from different levels cleanly
separated when a shared base directory is used.

| Flag | Description |
|---|---|
| `--year-start` | First year to download (≥ 1948) |
| `--year-end` | Last year to download (inclusive) |
| `--level` | FTP subdirectory, e.g. `pressure`, `surface` |
| `--variable` | Variable name as in filename, e.g. `air`, `slp`, `uwnd.sig995` |
| `--outdir` | Local output directory (created if absent) |
| `--threads N` | Override thread count (default: CPU core count) |
| `--skip-existing` | Skip files that already exist — safe to use for resuming |
| `--base-url URL` | Override root FTP URL |

Example — download all pressure-level air temperature files (1948–2024);
files land in `$SCRATCH/ncep/pressure/air.{year}.nc`:

```bash
python scripts/dload-ncep-data.py \
    --year-start 1948 --year-end 2024 \
    --level pressure --variable air \
    --outdir $SCRATCH/ncep \
    --skip-existing
```

## Building

```bash
make all       # standard build
make asan      # build with AddressSanitizer (produces pystarm_asan.*.so)
```

Set `MKLROOT` if not already set:
```bash
export MKLROOT=/global/common/software/nersc9/intel/oneapi/mkl/2024.1
```

Built and tested with GNU compiler on NERSC Perlmutter. Intel compiler build has unresolved errors.

## AddressSanitizer

To run with ASan:
1. `source scripts/asan-run-prep.sh`
2. `make asan`
3. In `soccer.py` (or any experiment script), uncomment the `importlib` block at the top to load `pystarm_asan.*.so` and comment out `import pystarm`.
4. Run and redirect output: `python soccer.py ... > asan_out.txt 2>&1`

See `known_issues/README.md` for known bugs and debugging notes.

## Known Issues

See `known_issues/README.md`. Key issue: `tsvdmi` with `mtype=eye` on soccer data crashes under multi-threaded execution. Does not crash single-threaded, with DCT, with tsvdmii, or on traffic data.

## NCEP Analysis Plans

### Analysis 1 — Extreme Event Preservation: EOF vs tsvdmii

**Motivation:** Climate scientists use EOF (Empirical Orthogonal Functions = matrix SVD/PCA) as the standard method for compressing/analyzing atmospheric data. A climate scientist noted that EOF tends to squash extreme weather events. This analysis tests whether tsvdmii preserves extremes better than EOF at the same reconstruction accuracy.

**Data:**
- Variable: air temperature (`air`)
- Year range: 1948–1957 (10 years, ~20 GB)
- All 17 pressure levels

**Method:**
- Both methods are run at the same relative error tolerance (`tol=0.01`)
- **tsvdmii:** run on full 4D tensor `(73, 144, 17, time)` with DCT transform, perm-mode `0123`. Implemented in `ncep-air_tsvdmii.py`.
- **EOF:** unfold full tensor to `(73×144×17, time)` = `(178416, 14612)`, apply randomized SVD (`sklearn.utils.extmath.randomized_svd`) at `k_max=1000`, determine effective rank `k*` from energy criterion `sum(s[:k*]²)/sum(s²) >= 1 - tol²`, truncate and reconstruct. Implemented in `ncep-air_eof.py`.
- **Extreme events:** defined per grid point `(lat, lon)` at 850 hPa as the top 0.5% (hot) and bottom 0.5% (cold) of original temperature values over time — 1% combined, ~146 time steps per grid point
- For each extreme time step at each grid point: compute pointwise relative error `|T_orig - T_reconst| / |T_orig|`, aggregate to **mean**, **max**, and **median**

**Output (shape 73×144 each):**
- Plotted (1×2 PDF): EOF vs tsvdmii relative error at extremes — shared color scale for direct comparison
- `--metric` controls which aggregate is plotted (mean/max/median, default: median)
- All three metrics saved to disk as `.npy` in the respective run directories
- Output PDF named `extremes_{level}hpa_{metric}.pdf`

**Scripts:**
| Script | Purpose |
|---|---|
| `scripts/ncep/ncep-air_tsvdmii.py` | Load NCEP data, compress+reconstruct with tsvdmii, save `reconstruction.npy` |
| `scripts/ncep/ncep-air_eof.py` | Load NCEP data, compress+reconstruct with EOF, save `reconstruction.npy` |
| `scripts/ncep/ncep-air_extremes.py` | Load both reconstructions, compute extreme errors, save maps + PDF |

**Output directory structure:**
```
$SCRATCH/pystarm/extremes/
    ncep-air_tsvdmii_0.01_dct_0123_{threads}/
        reconstruction.npy
        meta.json
        mean_err_850hpa.npy
        max_err_850hpa.npy
    ncep-air_eof_0.01_1000_{threads}/
        reconstruction.npy
        meta.json
        mean_err_850hpa.npy
        max_err_850hpa.npy
    extremes_850hpa.pdf
```

**Generate tsvdmii reconstruction:**
```bash
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export MKL_DYNAMIC=FALSE

python scripts/ncep/ncep-air_tsvdmii.py \
    --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
    --year-start 1948 --year-end 1957 \
    --tol 0.01 --mtype dct --perm-mode 0123 \
    --outdir /pscratch/sd/t/taufique/pystarm/extremes
```

**Generate EOF reconstruction:**
```bash
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export MKL_DYNAMIC=FALSE

python scripts/ncep/ncep-air_eof.py \
    --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
    --year-start 1948 --year-end 1957 \
    --tol 0.01 --k-max 1000 \
    --outdir /pscratch/sd/t/taufique/pystarm/extremes
```

**Generate extreme error maps and plots:**
```bash
python scripts/ncep/ncep-air_extremes.py \
    --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
    --year-start 1948 --year-end 1957 \
    --tsvdmii-dir /pscratch/sd/t/taufique/pystarm/extremes/ncep-air_tsvdmii_0.01_dct_0123_64 \
    --eof-dir     /pscratch/sd/t/taufique/pystarm/extremes/ncep-air_eof_0.01_1000_64 \
    --level-hpa 850 --metric median \
    --outdir /pscratch/sd/t/taufique/pystarm/extremes
```

**Output plots:** `extremes_850hpa_median.pdf` — 1×2 global maps (EOF vs tsvdmii median relative error at extremes), shared color scale, PlateCarree projection with coastlines. All three metric maps (mean/max/median) are saved as `.npy` in the respective run directories.

**Variables for extreme weather analysis:**

Selected based on standard atmospheric science practice for extreme weather studies using
pressure-level reanalysis data. See:
- NCEP/NCAR Reanalysis pressure-level data: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.pressure.html
- Atmospheric Reanalysis Overview, Climate Data Guide (UCAR): https://climatedataguide.ucar.edu/climate-data/atmospheric-reanalysis-overview-comparison-tables

| Variable | Level | Description | Extreme weather relevance |
|---|---|---|---|
| `air` | pressure (850 hPa) | Air temperature | Heat waves, cold snaps — **currently implemented** |
| `hgt` | pressure (500 hPa) | Geopotential height | Mid-troposphere blocking patterns driving persistent temperature extremes |
| `hgt` | pressure (850 hPa) | Geopotential height | Lower-troposphere circulation, cyclone tracking |
| `uwnd` / `vwnd` | pressure (850 hPa) | Wind components | Low-level jet streams, storm winds |
| `uwnd` / `vwnd` | pressure (300 hPa) | Wind components | Upper-level jet stream extremes |
| `slp` | surface | Sea level pressure | Cyclones, storms, pressure extremes |

**Future extensions:**
- Extend to 1979–2000 (22 years) once EOF scalability is confirmed
- Add `hgt` at 500 hPa as a second variable (blocking/circulation extremes)
- Add `slp` for storm/cyclone extremes
- Test at multiple pressure levels beyond 850 hPa

### Analysis 4 — Compression Quality on NCEP Data

**Status: data collected, plot scripts ready and updated.**

**Core idea:** Empirically compare compression quality across algorithms (tsvdmi, tsvdmii, EOF) and transforms (dct, eye) on NCEP air temperature data — confirming Theorem 17 (tsvdmii ≥ tsvdmi) and showing tsvdmii-dct outperforms EOF in a real-world setting.

**Series plotted:** tsvdmi-dct, tsvdmii-dct, tsvdmii-eye, EOF (all at perm_mode=0123, 64 threads).
EOF series is currently commented out in the plot script — uncomment to enable.

**Generate data:**
```bash
# Edit experiments.sh: set MTYPES=("dct" "eye"), loop over ncep-air
bash scripts/experiments.sh
# Parse logs into CSV (includes EOF runs)
python scripts/parse_logs.py $SCRATCH/pystarm/logs scripts/experiments.csv
```

**Generate plots:**
```bash
python scripts/plot_ncep-air_compression.py    # ncep-air: tsvdmi-dct, tsvdmii-dct, tsvdmii-eye, EOF
python scripts/plot_ncep-air_compression_4vs6.py  # ncep-air vs ncep-air-6 tsvdmii-dct comparison
```

**Output plots:** `plots/ncep-air_compression.pdf`, `plots/ncep-air_compression_4vs6.pdf`

**Notes on EOF:**
- EOF = randomized SVD applied to unfolded matrix `(73×144×17, time)` with k_max=1000
- At tol≥0.05 the k_max=1000 cap is hit — compression ratio saturates at 13507x, relative error at 0.033
- Tighter tolerances (0.0001, 0.001) show very low compression (~13-14x) — EOF struggles here
- Randomized vs exact SVD would give identical results for well-separated spectra

**Next steps:**
- Uncomment EOF series and verify tsvdmii-dct curve is clearly above EOF
- Quantify the gap at matching error levels

### Analysis 3 — DCT vs Identity Transform Comparison for NCEP Data

**Status: data collected, plot scripts ready.**

**Core idea:** Compare DCT transform vs identity (no transform, `eye`) for compressing NCEP air temperature data using tsvdmii. If DCT outperforms identity it confirms that the atmospheric data has strong spectral coherence across the transformed modes (level and time), which is physically expected.

**Generate data:**
```bash
# Edit experiments.sh: set MTYPES=("dct" "eye"), loop over ncep-air
bash scripts/experiments.sh
# Parse logs into CSV
python scripts/parse_logs.py $SCRATCH/pystarm/logs scripts/experiments.csv
```

**Generate per-slice rank data (complementary diagnostic):**
```bash
# dct ranks
python ncep_ranks.py -dname ncep-air -mtype dct -tol 0.01
# eye ranks
python ncep_ranks.py -dname ncep-air -mtype eye -tol 0.01
# Output saved to $SCRATCH/pystarm/ranks/ncep-air_tsvdmii_{tol}_{mtype}_0123_{threads}_ranks/
```

**Generate plots:**
```bash
python scripts/plot_ncep_dct_vs_eye.py    # compression curve comparison
python scripts/plot_ncep-air_ranks.py     # per-slice rank barplot: dct (top) vs eye (bottom)
```

**Output plots:** `plots/ncep_dct_vs_eye.pdf`, `plots/ncep-air_ranks.pdf`

**Next steps:**
- Run plots and verify DCT outperforms eye (higher compression at same error)
- Per-slice rank plot shows DCT concentrates energy — fewer, larger singular values per slice

### Analysis 2 — Higher-Order Tensor Decomposition via Time Mode Splitting

**Status: compression data collected at 64 threads; TTM scaling runs at 8/16/32 threads pending for tsvdmii-dct tol=0.01.**

**Core idea:** Instead of treating time as a single flat dimension, reshape it into multiple physically meaningful sub-dimensions `(tod=4, doy=365, year=10)` to create a 6-way tensor. The goal is to compare scalability breakdown between ncep-air (4-way) and ncep-air-6 (6-way) — does mode splitting change how time is spent across TTM, SVD, and reconstruction?

**Key observation (64 threads, tsvdmii-dct, tol=0.01):**

| | Mode 2 | Mode 3 | Mode 4 | Mode 5 | TTM Total | SVD Total |
|---|---|---|---|---|---|---|
| 4-way | 1.91s | 58.79s | — | — | 60.70s | — |
| 6-way | 1.89s | 1.37s | 1.76s | 1.11s | 6.14s | 4.31s |

Mode 3 in the 4-way case is the full time dimension (size 14612) — very expensive. In the 6-way case time is split into `(4, 365, 10)` — each sub-dimension is small, making all TTMs cheap. TTM and SVD are comparable in the 6-way case (6.14s vs 4.31s).

**Generate data:**
```bash
# Edit experiments.sh: set DNAME loop to ncep-air-6, NTHREADS to desired value (8, 16, 32, 64)
bash scripts/experiments.sh
# Parse logs into CSV
python scripts/parse_logs.py $SCRATCH/pystarm/logs scripts/experiments.csv
```

**Generate plots:**
```bash
python scripts/plot_ncep-air_compression_4vs6.py   # compression quality: 4-way vs 6-way
python scripts/plot_ncep-air_ttm_scaling_4vs6.py   # TTM strong scaling: 4-way vs 6-way stacked bars
```

**Output plots:** `plots/ncep-air_compression_4vs6.pdf`, `plots/ncep-air_ttm_scaling_4vs6.pdf`

**Pending:**
- ncep-air-6 tsvdmii-dct tol=0.01 scaling runs at 8, 16, 32 threads — submitted, awaiting results
- Once data lands: re-run `plot_ncep-air_ttm_scaling_4vs6.py` — missing bars fill in automatically

### Analysis 5 — SLP Extreme Event Preservation: EOF vs tsvdmii

**Motivation:** Same question as Analysis 1 but for sea level pressure (SLP). SLP is the primary indicator of tropical cyclones and mid-latitude storms — deep pressure minima (e.g. ~944 hPa for Cyclone Sidr) are the defining signature of extreme events. If compression smooths these minima, downstream storm detection and intensity analysis would be compromised. This analysis tests whether tsvdmii preserves SLP extremes better than EOF.

**Key events in the dataset (1985–2015, South/Southeast Asia focus):**
- **Cyclone Sidr** (Nov 2007): 944 hPa, Category 4, landfall Bangladesh — one of the most intense Bay of Bengal cyclones on record
- **Cyclone Aila** (May 2009): 967 hPa, severe cyclonic storm, Bangladesh/West Bengal

**Data:**
- Variable: sea level pressure (`slp`), surface level
- Year range: 1985–2015 (31 years)
- Data location: `/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/surface`
- Tensor shape: `(73, 144, 181056)` — no pressure level dimension

**Method:**
- Both methods run at `tol=0.01`
- **tsvdmii:** 3-way tensor `(lat, lon, time)`, `ttm_modes=[2]` (time mode only). Implemented in `ncep-slp_tsvdmii.py`.
- **EOF:** unfold to `(73×144, time)` = `(10512, 181056)`, randomized SVD at `k_max=1000`. Implemented in `ncep-slp_eof.py`.
- **Extreme events:** per grid point, bottom 0.5% (low pressure) and top 0.5% (high pressure) of time series — 1% combined
- Aggregate pointwise relative error `|orig - reconst| / |orig|` to mean, max, median

**Scripts:**
| Script | Purpose |
|---|---|
| `ncep-slp_tsvdmii.py` | Load SLP data, compress+reconstruct with tsvdmii, save `reconstruction.npy` |
| `ncep-slp_eof.py` | Load SLP data, compress+reconstruct with EOF, save `reconstruction.npy` |
| `ncep-slp_extremes.py` | Compute extreme errors, save maps + PDF |

**Output directory structure:**
```
$SCRATCH/pystarm/extremes/
    ncep-slp_tsvdmii_{tol}_{mtype}_{threads}/
        reconstruction.npy
        meta.json
        mean_err.npy
        max_err.npy
        median_err.npy
    ncep-slp_eof_{tol}_{k_max}_{threads}/
        reconstruction.npy
        meta.json
        mean_err.npy
        max_err.npy
        median_err.npy
    slp_extremes_{metric}.pdf
```

**Generate tsvdmii reconstruction:**
```bash
export OMP_NUM_THREADS=64; export MKL_NUM_THREADS=64; export MKL_DYNAMIC=FALSE

python ncep-slp_tsvdmii.py \
    --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/surface \
    --year-start 1985 --year-end 2015 \
    --tol 0.01 --mtype dct \
    --outdir /pscratch/sd/t/taufique/pystarm/extremes
```

**Generate EOF reconstruction:**
```bash
python ncep-slp_eof.py \
    --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/surface \
    --year-start 1985 --year-end 2015 \
    --tol 0.01 --k-max 1000 \
    --outdir /pscratch/sd/t/taufique/pystarm/extremes
```

**Generate extreme error maps and plot:**
```bash
python ncep-slp_extremes.py \
    --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/surface \
    --year-start 1985 --year-end 2015 \
    --tsvdmii-dir /pscratch/sd/t/taufique/pystarm/extremes/ncep-slp_tsvdmii_0.01_dct_64 \
    --eof-dir     /pscratch/sd/t/taufique/pystarm/extremes/ncep-slp_eof_0.01_1000_64 \
    --metric median \
    --outdir /pscratch/sd/t/taufique/pystarm/extremes
```

**Output plots:** `slp_extremes_median.pdf` — 1×2 global maps (EOF vs tsvdmii median relative error at extremes), shared color scale.

### Analysis 6 — Snapshot Visualization at Extreme Events

**Motivation:** Rather than global aggregate statistics, look at a specific known extreme event and visually compare the original field vs EOF reconstruction vs tsvdmii reconstruction at that exact time step. This makes reconstruction errors spatially interpretable — e.g. does tsvdmii preserve the pressure minimum of a cyclone better than EOF?

#### SLP snapshot — Cyclone Sidr

**Event:** Cyclone Sidr made landfall in Bangladesh on **November 15, 2007** at approximately 17:00 UTC. Closest NCEP 6-hourly step: **18:00 UTC**. Central pressure ~944 hPa.

**Region:** South/Southeast Asia — lat 5°–40°N, lon 60°–135°E

**Produces two PDFs:**
- `slp_snapshot_2007-11-15_18UTC.pdf` — 1×2 relative error maps (EOF vs tsvdmii), YlOrRd colormap
- `slp_values_2007-11-15_18UTC.pdf` — 1×3 normalized SLP (original / 1013.25, EOF / 1013.25, tsvdmii / 1013.25), RdBu_r colormap

**Command:**
```bash
python ncep-slp_snapshot.py \
    --data-dir   /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/surface \
    --year-start 1985 --year-end 2015 \
    --tsvdmii-dir /pscratch/sd/t/taufique/pystarm/extremes/ncep-slp_tsvdmii_0.01_dct_64 \
    --eof-dir     /pscratch/sd/t/taufique/pystarm/extremes/ncep-slp_eof_0.01_1000_64 \
    --outdir     /pscratch/sd/t/taufique/pystarm/extremes \
    --date 2007-11-15 --hour 18 \
    --lat-min 5 --lat-max 40 --lon-min 60 --lon-max 135
```

#### Air temperature snapshot — 1954 Midwest Heat Wave

**Event:** July 14, 1954 — record heat in Illinois (117°F / 47°C). Illinois is UTC-5 (CDT) in summer, so midday ≈ 17–18 UTC. Closest NCEP step: **18:00 UTC**. Analyzed at **850 hPa** (lower troposphere).

**Region:** Central/Eastern US — lat 25°–55°N, lon 220°–310°E (NCEP 0–360° convention)

**Produces two PDFs:**
- `air_snapshot_1954-07-14_18UTC.pdf` — 1×2 relative error maps (EOF vs tsvdmii), YlOrRd colormap
- `air_values_1954-07-14_18UTC.pdf` — 1×3 raw temperature in K (original, EOF, tsvdmii), RdBu_r colormap

**Command:**
```bash
python ncep-air_snapshot.py \
    --data-dir   /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
    --year-start 1948 --year-end 1957 \
    --tsvdmii-dir /pscratch/sd/t/taufique/pystarm/extremes/ncep-air_tsvdmii_0.01_dct_0123_64 \
    --eof-dir     /pscratch/sd/t/taufique/pystarm/extremes/ncep-air_eof_0.01_1000_64 \
    --outdir     /pscratch/sd/t/taufique/pystarm/extremes \
    --date 1954-07-14 --hour 18 \
    --level-hpa 850 \
    --lat-min 25 --lat-max 55 --lon-min 220 --lon-max 310
```

### Analysis 7 — EOF vs tsvdmii Compression Curve (Tolerance Sweep)

**Motivation:** Analysis 1 and 5 compare at a single tolerance (0.01). This analysis sweeps across tolerances to produce a full compression ratio vs relative error curve, showing how EOF and tsvdmii trade off compression against accuracy across the entire range.

**Method:** Run both `eof` and `tsvdmii` at tolerances `0.0001 0.001 0.01 0.025 0.05 0.1 0.25` via `experiments.py` / `experiments.sh`. Parse log files into CSV, plot with dedicated script.

**Run for ncep-air:**
```bash
export OMP_NUM_THREADS=64; export MKL_NUM_THREADS=64; export MKL_DYNAMIC=FALSE

for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25; do
    LOGFILE="$SCRATCH/pystarm/logs/ncep-air_eof_${TOL}_none_none_64"
    python experiments.py -alg eof -tol $TOL -k-max 1000 \
        -dname ncep-air \
        -dfile /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
        > $LOGFILE 2>&1
done
```

**Run for ncep-slp:** same pattern with `-dname ncep-slp` and the surface data path.

**Parse and plot:**
```bash
python scripts/parse_logs.py $SCRATCH/pystarm/logs scripts/experiments.csv
python scripts/plot_ncep_compression.py        # ncep-air: EOF vs tsvdmii
python scripts/plot_ncep_slp_compression.py    # ncep-slp: EOF vs tsvdmii
```

**Output plots:**
- `plots/ncep_compression.pdf` — ncep-air compression curve
- `plots/ncep_slp_compression.pdf` — ncep-slp compression curve

**Log file naming for EOF:** `{dname}_eof_{tol}_none_none_{threads}` (mtype/perm_mode fields set to `none` since EOF does not use them)

---

## Paper Todo List

### Paper Structure (agreed)

```
1. Intro
2. Prelim: tSVDM-I and tSVDM-II
3. Parallelization/Design
   - TTM: batched BLAS vs loop over multithreaded GEMM
   - tSVDM-I: parfor over singlethreaded LAPACK vs loop over multithreaded LAPACK
   - tSVDM-II: full/truncate vs SVD vals + re-run vs SVD vals + efficient re-run
   - 4-way vs 6-way TTM scaling (ncep-air, tsvdmii-dct, tol=0.01)  ← data ready
4. Applications
   - Climate (ncep-air):
       - Compression quality — 4-way (tsvdmi-dct, tsvdmii-dct, tsvdmii-eye, EOF)
       - Compression quality — 4-way vs 6-way (tsvdmii-dct)
       - Scaling — 6-way (tsvdmi vs tsvdmii, compress time only, wall time vs threads)
       - TTM scaling — 4-way vs 6-way
       - Per-slice rank (DCT vs eye, tsvdmii tol=0.01)
   - CFD:
       - Compression quality (tsvdmi-dct vs tsvdmii-dct)
       - Scaling (compress time only, wall time vs threads)
   - X-ray Crystallography:
       - Compression quality (tsvdmi-dct vs tsvdmii-dct)
       - Scaling (compress time only, wall time vs threads)
```

---

## Experiment Plan

### General conventions

- **Strong scaling thread counts:** 1, 2, 4, 8, 16, 32, 64 (doubling from 1)
- **Strong scaling parameters:** one fixed (tol, k) pair per dataset — do not sweep all values
  - ncep-air-6: tsvdmii tol=0.01, tsvdmi k=5
  - cfd: tsvdmii tol=0.1, tsvdmi k=20
  - crystallography: TBD, will mirror CFD setup
- **Datasets for benchmarks:** ncep-air-6, cfd, crystallography (when available)

---

### Category 1 — Parallel Performance

These experiments go in the Design/Implementation section of the paper.

#### Q1: Does batched GEMM outperform loop GEMM for TTM?

**Why it matters:** The paper claims batched GEMM is better than a loop over individual GEMMs, but we have no data to back this up. Three variants are implemented in `cpp/ops.cpp`:
- `ttm()` (batched): single `cblas_dgemm_batch_strided` call — MKL manages all blocks
- `ttm_loop()` (loop): serial loop over blocks, one `cblas_dgemm` per block, multi-threaded MKL per call
- `ttm_parfor()` (parfor): OMP parallel loop over blocks, each `cblas_dgemm` uses `mkl_set_num_threads_local(1)` — mirrors the slicewise SVD parallelism strategy

pybind11 bindings for all three are in `cpp/starm.cpp`. Rebuild required after any C++ changes: `make all`.

**To collect data:**
```bash
sbatch scripts/job_nersc_benchmark_ttm.sh
# or directly:
bash scripts/benchmark_ttm.sh
```

**Output:** `scripts/benchmark_ttm.csv` with columns `dname, mode, ttm_variant, threads, run_id, time_sec`
- Upsert behavior: safe to re-run; writes sentinel `-1` before each run and overwrites with actual time on success
- Datasets: ncep-air, ncep-air-6, cfd; thread counts 64→1; 5 runs each
- `ttm_variant` values: `batched`, `loop`, `parfor`

---

#### Q2: Does parfor+sequential SVD outperform loop+parallel-MKL SVD?

**Why it matters:** The paper describes two parallelization strategies for slicewise SVD:
- **parfor+sequential** (`slicewise_svd`): OMP parallel loop over slices, each `dgesvd` call uses 1 MKL thread
- **sequential+MKL** (`slicewise_svd_seq`): sequential loop over slices, each `dgesvd` call uses all MKL threads

**C++ implementation:**
- `mkl_set_num_threads_local(1)` added to all 4 parallel slicewise SVD functions in `cpp/ops.cpp`
- `slicewise_svd_seq` added to `cpp/ops.cpp` with comment explaining its purpose
- pybind11 binding added in `cpp/starm.cpp`
- Rebuild required: `make all`

**To collect data:**
```bash
sbatch scripts/job_nersc_benchmark_svd.sh
# or directly:
bash scripts/benchmark_svd.sh
```

**Output:** `scripts/benchmark_svd.csv` with columns `dname, svd_variant, threads, run_id, time_sec`
- Variants: `parfor` vs `seq`
- Datasets: ncep-air-6, cfd; thread counts 64→1; 5 runs each
- TTM pre-applied before benchmarking so input matches real compress pipeline
- Upsert behavior: safe to re-run; writes sentinel `-1` before each run

---

#### Q3: Which tSVDM-II strategy is most efficient?

**Why it matters:** The paper describes 3 strategies for tSVDM-II. Only strategy 2 (SVDvals first + targeted recompute) is currently implemented and used. Strategies 1 (full decompose then truncate) and 3 (SVDvals + memory-efficient storage) need to be implemented and benchmarked.

**Experiments needed:**
- Run tsvdmii compress at 1, 2, 4, 8, 16, 32, 64 threads with each strategy
- Fixed parameters per dataset
- Log total compress time and breakdown (SVDvals, SVDks)
- Datasets: ncep-air-6, cfd (crystallography when available)

**Status:** Strategy 2 implemented (current default). Strategy 1 and 3 need C++ implementation. Note: paper text in `03_algorithms.tex` incorrectly says strategy 1 is the current implementation — needs correction. Deferred to a future session.

---

### Category 2 — Application

**Note:** Each dataset section in the paper needs a visualization of the actual data (e.g. a representative snapshot or slice). Brainstorm and design these once the CFD and crystallography dataset details are known.

These experiments go in the Application section of the paper.

#### Q4: Does tensor compression beat matrix compression (EOF)?

**Experiments:** tsvdmi-dct, tsvdmii-dct, tsvdmii-eye, EOF on ncep-air — compression ratio vs relative error curve.
**Status:** Data collected. Plot ready (`plots/ncep-air_compression.pdf`). ✅

---

#### Q5: Does tsvdmii beat tsvdmi?

**Experiments:** Same compression quality sweep as Q4; also cfd compression quality.
**Status:** Data collected. Plots ready for ncep-air and cfd. ✅

---

#### Q6: Does DCT transform help vs identity?

**Experiments:** tsvdmii-dct vs tsvdmii-eye on ncep-air — compression curve + per-slice rank distribution.
**Status:** Data collected. Plots ready (`plots/ncep-air_compression.pdf`, `plots/ncep-air_ranks.pdf`). ✅

---

#### Q7: Does 6-way decomposition improve TTM scalability without hurting compression?

Two sub-questions with separate data sources:

**Q7a — Compression quality (4-way vs 6-way):**
- Run via `experiments.py` as today, all tol values, 64 threads, tsvdmii-dct
- Datasets: ncep-air (perm_mode=0123) and ncep-air-6 (perm_mode=012345)
- **Status:** Data collected, plot ready (`plots/ncep-air_compression_4vs6.pdf`) ✅

**Q7b — TTM scaling (4-way vs 6-way):**
- Uses same `benchmark_ttm.sh` as Q1 — ncep-air (4-way) and ncep-air-6 (6-way) both included
- Per-mode TTM times captured naturally by the CSV structure (one row per mode per run)
- Thread counts: 64→1; 5 runs each

**To collect data:** same run as Q1 — `bash scripts/benchmark_ttm.sh`

---

#### Q8: How does the implementation scale with thread count?

**Experiments:** Strong scaling at 1, 2, 4, 8, 16, 32, 64 threads, fixed parameters per dataset.
- ncep-air-6: tsvdmii tol=0.01, tsvdmi k=5
- cfd: tsvdmii tol=0.1, tsvdmi k=20
- crystallography: TBD, mirror CFD setup

**Multiple runs:** Run each configuration 5 times. Log files named with `_run{i}` suffix (e.g. `ncep-air-6_tsvdmii_0.01_dct_012345_64_run1`). `parse_logs.py` extracts `run_id` from filename automatically.

**To collect data:**
1. Edit `experiments.sh`: set DNAME loop to `"ncep-air" "ncep-air-6" "cfd"`, set ALG loop to `"tsvdmi" "tsvdmii"`, set fixed k/tol per dataset (see General conventions above), loop over NTHREADS 1 2 4 8 16 32 64
2. Run:
```bash
sbatch scripts/job_nersc_experiments.sh
# or directly:
bash scripts/experiments.sh
```
3. Parse logs:
```bash
python scripts/parse_logs.py $SCRATCH/pystarm/logs scripts/experiments.csv
```

**Note:** Existing data at 8/16/32/64 threads is single-run only. Full 5-run data is needed at all 7 thread counts.

---

### Stretch Goals

- [ ] Extreme event analysis: ncep-air 850hPa EOF vs tsvdmii (Analysis 1)
- [ ] SLP extreme events + Cyclone Sidr snapshot (Analysis 5, 6)

---

## Data Visualization

### ncep-air 3D Seasonal Snapshot

**Script:** `scripts/plot_ncep-air_3d_seasons.py`

**Purpose:** 3D stacked pressure-level visualization of NCEP air temperature at the four seasonal snapshots (March equinox, June solstice, September equinox, December solstice) for a given year. Shows all 17 pressure levels as stacked translucent planes with temperature in °C (RdBu_r, centered at 0°C). Bottom level (1000 hPa) is fully opaque; upper levels fade out.

**Command:**
```bash
python scripts/plot_ncep-air_3d_seasons.py \
    --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
    --year 1952 \
    --outdir plots/
```

**Output:** `plots/ncep-air_3d_seasons_{year}.pdf` — 1×4 figure, one panel per season.

**Known issue:** Coastline overlay not working correctly — to be fixed in a future session.

---

## NERSC Job Scripts

Slurm batch scripts for running experiments on Perlmutter (account: m4293, regular QOS, 1 CPU node).

| Script | Wall time | Runs |
|---|---|---|
| `scripts/job_nersc_benchmark_ttm.sh` | 23:00:00 | `benchmark_ttm.sh` |
| `scripts/job_nersc_benchmark_svd.sh` | 23:00:00 | `benchmark_svd.sh` |
| `scripts/job_nersc_experiments.sh`   | 04:00:00 | `experiments.sh` |

Submit with:
```bash
sbatch scripts/job_nersc_benchmark_ttm.sh
sbatch scripts/job_nersc_benchmark_svd.sh
sbatch scripts/job_nersc_experiments.sh
```
