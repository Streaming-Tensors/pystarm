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
experiments.py          Unified experiment script for all datasets
scripts/
  experiments.sh        Batch-run experiments (Python and/or MATLAB); set RUN_PYTHON/RUN_MATLAB switches at top
  parse_logs.py         Parses experiment log files into experiments.csv
  hosvd_experiment.m    MATLAB HOSVD experiment script (single run, called by experiments.sh)
  plot_alg-compare.py   Plot relative error vs compression ratio for a given dataset
  asan-run-prep.sh      Sets up environment for running with AddressSanitizer
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
| ncep-air       | .nc (NetCDF)  | 73 × 144 × 17 × T (per yr) | No  | `read_ncep_air` (ncep.py)| —             |

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
matching the convention in `experiments.py`.  For 11 years (1948–1958) the tensor
shape is roughly **73 × 144 × 17 × 16071** (~1.1 GB as float64).

Reading is done with `scripts/ncep.py` (settings currently hard-coded at the top of
that file).  The function to call is `read_ncep_air(data_dir, variable, year_start,
year_end)`.

## experiments.py

Unified script for running experiments on any dataset.

```bash
python experiments.py -alg <tsvdmi|tsvdmii> -mtype <dct|eye|hosvd> \
    -k <rank> -tol <tolerance> \
    -dname <soccer|traffic-color|traffic-gray|dcmall|cfd> -dfile <path> \
    -perm-mode <digit string e.g. 0123>
```

- `-perm-mode`: digit string passed to `np.transpose` to reorder modes so transformation modes are last. e.g. `"0123"` → `(0,1,2,3)`.
- `ttm_modes = list(range(2, arr.ndim))` — transformation is applied on all modes except the first two.
- After permutation, a Fortran-order copy is made slice-by-slice along the last axis.
- For `mtype=eye`, no TTM is applied (identity transform is a no-op).
- For `mtype=hosvd`, HOSVD is run once via pyttb to get all factor matrices, then applied per mode.

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
- Both methods are run at the same relative error tolerance
- EOF: unfold full tensor to `(73×144×17, 14600)`, apply randomized SVD (`sklearn.utils.extmath.randomized_svd`), determine rank k from tolerance using energy criterion (same as tsvdmii), reconstruct
- tsvdmii: run on full tensor with matching tolerance `tol`, reconstruct
- Extreme events: defined per grid point as top 1% (99th percentile) of temperature values over time at 850 hPa (near-surface level)
- For each grid point `(lat, lon)`: collect all ~146 extreme time steps, compute relative reconstruction error at each, aggregate to **mean** and **max**

**Output (4 global maps):**
- EOF — mean relative error at extreme grid points
- EOF — max relative error at extreme grid points
- tsvdmii — mean relative error at extreme grid points
- tsvdmii — max relative error at extreme grid points

**Generate data:**
```bash
# tsvdmii data already available from existing experiments
# EOF pipeline needs to be run separately via ncep_extremes.py (not yet implemented)
```

**Generate plots:**
```bash
# Not yet implemented — will be added to ncep_extremes.py
# Output will go to $SCRATCH/pystarm/extremes/
# Copy final maps to plots/ manually
```

**Output plots:** 4 global maps (EOF mean, EOF max, tsvdmii mean, tsvdmii max relative error at extremes)

**Implementation notes (to be decided when implementing):**
- New standalone script `ncep_extremes.py`
- Save intermediate results (compressed representations, extreme masks) to avoid rerunning expensive steps
- Output to `$SCRATCH/pystarm/extremes/{run_name}/`
- Use cartopy for geographic map projections (coastlines etc.)

**Future extensions:**
- Extend to 1979–2000 (22 years) once EOF scalability is resolved
- Add geopotential height (`hgt`) as a second variable
- Test at multiple pressure levels beyond 850 hPa

### Analysis 4 — tsvdmi vs tsvdmii Compression Quality on NCEP Data

**Status: data collected, plot script ready.**

**Core idea:** Empirically verify on real NCEP atmospheric data that tsvdmii achieves better or equal compression than tsvdmi at the same error level — confirming Theorem 17 from the paper in a real-world setting.

**Generate data:**
```bash
# Edit experiments.sh: set MTYPES=("dct"), loop over ncep-air and ncep-air-6
bash scripts/experiments.sh
# Parse logs into CSV
python scripts/parse_logs.py $SCRATCH/pystarm/logs scripts/experiments.csv
```

**Generate plots:**
```bash
cd /path/to/pystarm
python scripts/plot_ncep_compression.py    # ncep-air
python scripts/plot_ncep6_compression.py   # ncep-air-6
```

**Output plots:** `plots/ncep_compression.pdf`, `plots/ncep6_compression.pdf`

**Next steps:**
- Run plots and verify tsvdmii curve is clearly above tsvdmi (higher compression at same error)
- Quantify the gap

### Analysis 3 — DCT vs Identity Transform Comparison for NCEP Data

**Status: data collected, plot script ready.**

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
# Copy interesting plots to plots/ manually
```

**Generate plots:**
```bash
python scripts/plot_ncep_dct_vs_eye.py
```

**Output plots:** `plots/ncep_dct_vs_eye.pdf`

**Next steps:**
- Run plot and verify DCT outperforms eye
- Compare per-slice rank histograms/barplots for dct vs eye from scratch directory

### Analysis 2 — Higher-Order Tensor Decomposition via Time Mode Splitting

**Status: compression data collected at 64 threads only, scaling data pending.**

**Core idea:** Instead of treating time as a single flat dimension, reshape it into multiple physically meaningful sub-dimensions `(tod=4, doy=365, year=10)` to create a 6-way tensor. The goal is to compare scalability breakdown between ncep-air (4-way) and ncep-air-6 (6-way) — does mode splitting change how time is spent across TTM, SVD, and reconstruction?

**Generate data:**
```bash
# Edit experiments.sh: set DNAME loop to ncep-air-6, NTHREADS to desired value (8, 16, 32, 64)
bash scripts/experiments.sh
# Parse logs into CSV
python scripts/parse_logs.py $SCRATCH/pystarm/logs scripts/experiments.csv
```

**Generate plots:**
```bash
python scripts/plot_ncep6_compression.py   # compression quality
python scripts/plot_ncep6_scaling.py       # scaling breakdown (requires data at multiple thread counts)
```

**Output plots:** `plots/ncep6_compression.pdf`, `plots/ncep6_scaling.pdf`

**Pending:**
- ncep-air-6 scaling experiments at 8, 16, 32 threads — currently running
- Once data is collected: compare scaling stacked bar charts for ncep-air vs ncep-air-6 side by side
- May need a new combined plot script for the side-by-side comparison
