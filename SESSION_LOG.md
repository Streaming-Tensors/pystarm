# Session Log

---

## Session — 2026-04-18

### What was done

#### C++ changes (requires `make all` before running)
- `cpp/ops.cpp`: Added `mkl_set_num_threads_local(1)` / `mkl_set_num_threads_local(0)` to all 4 parallel slicewise SVD functions (`slicewise_svd`, `slicewise_svdx`, `slicewise_svdvals`, `slicewise_svdks`) to match NERSC version
- `cpp/ops.cpp`: Added `slicewise_svd_seq` — sequential-loop variant for benchmarking (no OMP, each `dgesvd` uses all MKL threads). Comment at top of function explains the design.
- `cpp/starm.cpp`: Added pybind11 binding for `slicewise_svd_seq`

#### New scripts
- `scripts/benchmark_ttm.py` — benchmarks `pystarm.ttm` (batched) vs `pystarm.ttm_loop` (serial) per mode per dataset
- `scripts/benchmark_ttm.sh` — bash driver; loops over datasets and thread counts 64→1
- `scripts/benchmark_svd.py` — benchmarks `slicewise_svd` (parfor) vs `slicewise_svd_seq` (sequential MKL) per dataset
- `scripts/benchmark_svd.sh` — bash driver; loops over datasets and thread counts 64→1
- `scripts/job_nersc_benchmark_ttm.sh` — Slurm job for benchmark_ttm.sh (23h, m4293, regular, CPU)
- `scripts/job_nersc_benchmark_svd.sh` — Slurm job for benchmark_svd.sh (23h, m4293, regular, CPU)
- `scripts/job_nersc_experiments.sh` — Slurm job for experiments.sh (4h, m4293, regular, CPU)
- `scripts/plot_ncep-air_3d_seasons.py` — 3D stacked pressure-level visualization for seasonal snapshots

#### Updated scripts
- `scripts/experiments.sh`: Added `NUM_RUNS=5` and `_run{RUN_ID}` suffix on all log filenames
- `scripts/parse_logs.py`: Updated `parse_filename` to extract `run_id` from `_run{N}` suffix; added `run_id` to COLUMNS

---

### Experiment status after this session

| Q | Question | Status |
|---|---|---|
| Q1 | Batched GEMM vs loop GEMM for TTM | Scripts ready ✅ — data not collected ⬜ |
| Q2 | parfor+sequential SVD vs sequential+MKL SVD | Scripts + C++ ready ✅ — data not collected ⬜ |
| Q3 | tSVDM-II strategies 1 & 3 | Deferred — C++ not implemented ⬜ |
| Q4 | Tensor vs matrix (EOF) compression | Data collected, plot ready ✅ |
| Q5 | tsvdmii vs tsvdmi | Data collected, plot ready ✅ |
| Q6 | DCT vs identity transform | Data collected, plot ready ✅ |
| Q7a | 4-way vs 6-way compression quality | Data collected, plot ready ✅ |
| Q7b | 4-way vs 6-way TTM scaling | Scripts ready ✅ — data not collected ⬜ |
| Q8 | Strong scaling (1–64 threads) | Scripts ready ✅ — need re-runs at 1,2,4 threads with 5 runs each ⬜ |

### Known issues
- `scripts/plot_ncep-air_3d_seasons.py`: Coastline overlay not rendering correctly — to be fixed in a future session
- `scripts/benchmark_ttm.py`: Segfault on mode 3 (time dimension, size 14612) at some thread counts — likely OOM. Rows with `time_sec=-1` in CSV indicate crashed runs.

### Next steps
- Run `make all` on NERSC before submitting any jobs
- Submit benchmark jobs: `sbatch scripts/job_nersc_benchmark_ttm.sh`, `sbatch scripts/job_nersc_benchmark_svd.sh`
- For Q8: manually edit `experiments.sh` to set correct DNAME loop, ALG loop, and fixed k/tol per dataset, then `sbatch scripts/job_nersc_experiments.sh`
- Fix coastline rendering in `plot_ncep-air_3d_seasons.py`
- Write plotting scripts for benchmark_ttm.csv and benchmark_svd.csv once data is collected
