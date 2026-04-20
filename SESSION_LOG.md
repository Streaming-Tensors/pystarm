# Session Log

---

## Session — 2026-04-20

### What was done

#### Benchmark data collected
- Ran `benchmark_ttm.sh` and `benchmark_svd.sh` on NERSC — data now in `scripts/benchmark_ttm.csv` and `scripts/benchmark_svd.csv`

#### SVD benchmark data analysis
- **parfor** scales perfectly from 1→64 threads (time halves with every thread doubling) on both cfd and ncep-air-6
- **seq** degrades as threads increase — at 64 threads it is ~550× slower than parfor for ncep-air-6, ~84× slower for cfd
- At 1 thread both variants are identical (as expected)
- Degradation of seq is a well-known MKL behavior: for small matrices (73×144), thread spin-up and per-call memory allocation overhead dominates over actual computation. More threads → more idle thread spinning → worse total time
- The degradation pattern is **not well documented in literature** for the slicewise setting — could be a paper contribution
- Note: SVD benchmarks should be re-run on a dataset with larger slice matrix dimensions to further characterize the behavior
- References:
  - [GESVD is slow on small matrix - Intel Community](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/GESVD-is-slow-on-small-matrix-compared-to-numerical-recipes/td-p/882288)
  - [Large overhead and spin time in MKL functions - Intel Community](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/Large-overhead-and-spin-time-reported-in-MKL-functions/td-p/971581)
  - [MKL Compact Matrix Functions - Inside HPC](https://insidehpc.com/2018/02/intel-mkl-compact-matrix-functions-attain-significant-speedups/)
  - [OpenBLAS dsytrd multi-thread degradation](https://github.com/OpenMathLib/OpenBLAS/issues/3801)
  - [STFC LAPACK benchmark study](https://epubs.stfc.ac.uk/manifestation/1885/DLTR-2007-005.pdf)

#### TTM benchmark data analysis
- Batched GEMM clearly outperforms loop for mode 2 at high thread counts (e.g. ~11× faster for ncep-air mode 2 at 64 threads)
- For large dimensions (ncep-air mode 3, time dim 14612) batched and loop are essentially tied — both reduce to a single large GEMM
- For cfd (small dimensions) batched and loop are essentially identical across all thread counts
- ncep-air mode 3 at 1 thread missing (-1) — likely wall time exhaustion, not OOM
- 4-way vs 6-way TTM scaling clearly visible: mode 3 in 4-way is very expensive; in 6-way time is split into small dims making all TTMs cheap

#### C++ diagnostic prints added then commented out
- `cpp/ops.cpp` `slicewise_svd_seq`: added `mkl_get_max_threads` / `mkl_domain_get_max_threads(MKL_DOMAIN_LAPACK)` print and per-slice timing — now commented out
- `cpp/ops.cpp` `slicewise_svd`: added per-slice timing — now commented out
- Requires `make all` to take effect

#### Plot changes
- `ncep-air_extremes.py`: updated `plot_maps` for single ACM column format — 2×1 stacked layout, vertical colorbar on right, figure size 3.33×4.0 inches, reduced font sizes, dpi=300

#### Centering changes attempted and reverted
- Attempted to add time-mean centering to `ncep-air_tsvdmii.py` and `ncep-air_eof.py` before compression and add mean back after reconstruction
- Reverted due to OOM crash on NERSC — the ~20GB tensor leaves insufficient headroom
- **TODO:** revisit centering — implement memory-efficient slice-by-slice approach

#### Experiment readiness assessment
| Q | Status |
|---|---|
| Q1 | Data sufficient for conclusions ✅ |
| Q2 | Data sufficient for conclusions ✅ |
| Q3 | Not started ❌ |
| Q4 | Data sufficient ✅ |
| Q5 | Data sufficient ✅ |
| Q6 | Data sufficient ✅ |
| Q7a | Data sufficient ✅ |
| Q7b | Data sufficient ✅ |
| Q8 | Partial — missing 1/2/4 thread runs with 5 repeats ⚠️ |
| Analysis 1 (air extremes) | Reconstructions running — pending ⚠️ |

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
