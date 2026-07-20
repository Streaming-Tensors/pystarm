# Paper reproduction

This directory contains the scripts and data used to produce the results in
our ICPP 2026 paper (https://arxiv.org/abs/2605.16058).

## Running the scripts

All scripts in this directory expect to be run from the `paper/` directory, not
from the repo root. For example:

```bash
cd paper
python experiments.py -alg tsvdmii -mtype dct -tol 0.01 -dname cfd -dfile /path/to/cfd
```

## Datasets

| dname          | Format        | Shape                       | Normalized | Perm modes    |
|----------------|---------------|-----------------------------|------------|---------------|
| soccer         | .mp4 (video)  | H × W × T                   | No         | 012           |
| traffic-color  | .bin (binary) | H × W × 3 × T               | Yes        | 0123, 0321    |
| traffic-gray   | .bin (binary) | H × W × T                   | Yes        | 012, 021, 120 |
| dcmall         | .tif (image)  | 191 × 1280 × 307            | No         | 021           |
| cfd            | .h5 (HDF5)    | 253 × 253 × 253 × 4 × 6     | No         | 01234         |
| ncep-air       | .nc (NetCDF)  | 73 × 144 × 17 × T (per yr)  | No         | 0123          |
| ncep-air-6     | .nc (NetCDF)  | 73 × 144 × 17 × 4 × 365 × Y | No         | 012345        |
| xray           | .npy (numpy)  | 300 × 400 × 400             | No         | 012           |

Normalized datasets (traffic-color, traffic-gray) are divided by their Frobenius
norm before compression. `traffic-gray` is derived from `traffic.bin` using
BT.601 luminance weights matching MATLAB's `im2gray`.

### traffic.bin binary format

Header: `uint32 height`, `uint32 width`, `float64 fps`.
Body: float64 values in column-major order, shape `(H, W, 3, T)`.

### NCEP air temperature dimensions

Each annual NetCDF file (`air.{year}.nc`) has these axes as read by xarray:

| Axis | Name  | Size         | Description                                |
|------|-------|--------------|--------------------------------------------|
| 0    | time  | 1460 or 1464 | 6-hourly observations (1464 in leap years) |
| 1    | level | 17           | Pressure levels (hPa)                      |
| 2    | lat   | 73           | 90°N → 90°S, 2.5° spacing                  |
| 3    | lon   | 144          | 0° → 357.5°E, 2.5° spacing                 |

Pressure levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100,
70, 50, 30, 20, 10 hPa.

`read_ncep_air` in `experiments.py` transposes to `(lat, lon, level, time)` so
spatial modes come first. For 10 years (1948–1957) the tensor shape is
`73 × 144 × 17 × 14612` (~19.46 GB as float64).

## Downloading NCEP data

`scripts/dload-ncep-data.py` is a multi-threaded downloader for NCEP Reanalysis
NetCDF files from NOAA's FTP server. Example — download all pressure-level air
temperature files for 1948–2024:

```bash
python scripts/dload-ncep-data.py \
    --year-start 1948 --year-end 2024 \
    --level pressure --variable air \
    --outdir /path/to/ncep --skip-existing
```

Files land in `<outdir>/pressure/{variable}.{year}.nc`. Run with `--help` for
full usage. See NOAA PSL for the full variable list:
https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.pressure.html

## Running experiments

### experiments.py — single experiment

Unified runner supporting all algorithms and datasets:

```bash
python experiments.py -alg <tsvdmi|tsvdmii|eof> -mtype <dct|eye|hosvd> \
    -k <rank> -tol <tolerance> -k-max <max_svd_rank> \
    -dname <soccer|traffic-color|traffic-gray|dcmall|cfd|ncep-air|ncep-air-6|xray> \
    -dfile <path> -perm-mode <digit string e.g. 0123>
```

- `-perm-mode`: digit string passed to `np.transpose` to reorder modes. The
  transformation is applied to all modes except the first two.
- For `mtype=eye`, no TTM is applied (identity transform).
- For `alg=eof`, `-mtype` and `-perm-mode` are ignored; `-k-max` sets the max
  SVD rank (default 1000).

### experiments.sh — batch driver

Sweeps `experiments.py` over datasets, algorithms, tolerances/ranks, and thread
counts. Edit the switches at the top of the file:

- `MACHINE` — set to `nersc-perlmutter-cpu` or `alcf-aurora`
- `RUN_PYTHON` / `RUN_MATLAB` — toggle which experiments to run
- `DNAME` loop and `K_VALUES` / `PERM_MODES` — select datasets and parameters
- `THREADS` — thread counts to sweep

```bash
bash scripts/experiments.sh
```

Each run produces one log file per parameter combination in `$OUTPUT_DIR`
named `{dname}_{alg}_{k_or_tol}_{mtype}_{perm_mode}_{threads}_run{i}`.

## Parsing logs

Consolidate log files into a single CSV:

```bash
python scripts/parse_logs.py <logdir> <outfile.csv>
```

## Generating plots

All plot scripts read from a CSV file at a fixed path inside `scripts/` (edit
the `CSV_FILE` variable at the top of each script if needed) and write PDFs
to `plots/`.

```bash
python scripts/plot_ncep-air_compression.py
python scripts/plot_cfd_compression.py
python scripts/plot_xray_compression.py
# ... etc.
```

## Reproducing paper analyses

### Extreme event preservation (EOF vs tsvdmii)

Pipeline that compresses NCEP air temperature data with tsvdmii and EOF at
matching tolerances, then compares reconstruction quality at extreme
temperature events (top/bottom 0.5% per grid point):

```bash
bash scripts/ncep/ncep-air_extreme.sh
```

This runs three steps in sequence:
1. `scripts/ncep/ncep-air_tsvdmii.py` — tsvdmii compression + reconstruction
2. `scripts/ncep/ncep-air_eof.py` — EOF compression + reconstruction
3. `scripts/ncep/ncep-air_extremes.py` — extreme error maps + cartopy plots

Edit the machine-specific paths and parameters at the top of the script before
running.

### DCT vs identity transform

Compares tsvdmii-dct vs tsvdmii-eye (no transform) on NCEP air temperature.
Set `MTYPES=("dct" "eye")` in `scripts/experiments.sh`, then:

```bash
bash scripts/experiments.sh
python scripts/parse_logs.py $OUTPUT_DIR scripts/experiments_<machine>.csv
python scripts/plot_ncep_dct_vs_eye.py
```

### Compression quality

Compression ratio vs relative error across algorithms (tsvdmi, tsvdmii, EOF)
and datasets:

```bash
bash scripts/experiments.sh
python scripts/parse_logs.py $OUTPUT_DIR scripts/experiments_<machine>.csv
python scripts/plot_ncep-air_compression.py
python scripts/plot_cfd_compression.py
python scripts/plot_xray_compression.py
```

## Scripts overview

### Experiment runners

- `experiments.py` — unified experiment runner supporting all algorithms
  (tsvdmi, tsvdmii, eof) and datasets (cfd, ncep-air, ncep-air-6, xray).
- `scripts/experiments.sh` — bash driver that sweeps over datasets,
  algorithms, tolerances/ranks, and thread counts.
- `scripts/ncep/ncep-air_tsvdmii.py`, `scripts/ncep/ncep-air_eof.py` —
  compress NCEP air data with tsvdmii and EOF respectively, save reconstruction.
- `scripts/ncep/ncep-air_extremes.py` — compare EOF vs tsvdmii reconstruction
  quality at extreme weather events, produce cartopy plots.
- `scripts/ncep/ncep-air_extreme.sh` — pipeline driver running the three
  ncep-air extreme event analysis scripts end to end.

### Benchmarks

- `scripts/benchmark_ttm.py` — benchmarks the three TTM parallelization
  variants (batched, loop, parfor) per mode per dataset.
- `scripts/benchmark_svd.py` — benchmarks the two slicewise SVD strategies
  (parfor+sequential-MKL vs sequential-loop+MKL).
- `scripts/benchmark_ttm.sh`, `scripts/benchmark_svd.sh` — bash drivers for
  the above.

### Job submission

- `scripts/job_nersc_*.sh` — Slurm job scripts for NERSC Perlmutter.
- `scripts/job_alcf_*.sh` — PBS job scripts for ALCF Aurora.

### Data utilities

- `scripts/dload-ncep-data.py` — multi-threaded download script for NCEP
  Reanalysis NetCDF files from NOAA FTP.
- `scripts/ncep.py` — standalone NCEP data reader (settings hard-coded at top).
- `scripts/parse_logs.py` — parses experiment log files into a single CSV.
- `scripts/hosvd_experiment.m`, `scripts/traffic_reader.m`,
  `scripts/traffic_writer.m` — MATLAB helpers for HOSVD and traffic data.

### Environment setup

- `scripts/nersc-env-setup.sh`, `scripts/alcf-env-setup.sh` — module loads
  and environment variables for each machine.
- `scripts/asan-run-prep.sh` — sets up environment for running with
  AddressSanitizer.

### Plot scripts

All plot scripts read from CSV files produced by `parse_logs.py` or the
benchmark scripts, and write PDFs to `plots/`.

- `scripts/plot_benchmark_ttm.py`, `scripts/plot_benchmark_svd.py` — TTM and
  SVD benchmark results.
- `scripts/plot_cfd_compression.py`, `scripts/plot_cfd_scaling.py` — CFD
  compression quality and strong scaling.
- `scripts/plot_xray_compression.py`, `scripts/plot_xray_scaling.py` — X-ray
  crystallography compression quality and strong scaling.
- `scripts/plot_ncep-air_compression.py`,
  `scripts/plot_ncep-air_compression_4vs6.py` — NCEP air compression quality
  (4-way vs 6-way).
- `scripts/plot_ncep-air_ttm_scaling.py`,
  `scripts/plot_ncep-air_ttm_scaling_4vs6.py`,
  `scripts/plot_ncep-air-4_scaling.py`, `scripts/plot_ncep_scaling.py`,
  `scripts/plot_ncep6_scaling.py`, `scripts/plot_ncep6_compression.py` —
  NCEP air scaling and compression plots.
- `scripts/plot_ncep_dct_vs_eye.py` — NCEP air compression comparing DCT
  vs identity transform.
