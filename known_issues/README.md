# Known Issues

> **Note:** All commands below assume you are running from the `paper/`
> directory (i.e., `cd paper` first).

## 1. Crash with tsvdmi, soccer data, identity transformation, and multiple threads

### Description

Running `tsvdmi` on the soccer dataset with the identity transformation crashes when multiple threads are active. The crash does not occur when running single-threaded (with both OpenMP and MKL thread counts set to 1), when using a different matrix type such as DCT, when using `tsvdmii`, or when running on other datasets such as traffic data.

### Reproduction

The crash occurs with the following command:

```bash
python soccer.py -alg tsvdmi -k 50 -mtype eye
```

The issue is **not observed** in these configurations:

- Single-threaded execution (set both OpenMP and MKL thread counts to 1):
  ```bash
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python soccer.py -alg tsvdmi -k 50 -mtype eye
  ```
- Different transformation matrix type (e.g., DCT):
  ```bash
  python soccer.py -alg tsvdmi -k 50 -mtype dct
  ```
- Different algorithm (tsvdmii):
  ```bash
  python soccer.py -alg tsvdmii -tol 0.1 -mtype eye
  ```
- Different datasets (traffic data does not exhibit this crash).

**Additional investigation:** Since the issue arises in parallel runs, one suspicion was that it could be related to setting both `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to the number of available cores, causing nested parallelism. To verify, two approaches were tried to restrict MKL to a single thread per OpenMP worker:

1. Setting `MKL_NUM_THREADS=1` via environment variable.
2. Calling `mkl_set_num_threads_local(1)` inside the `#pragma omp parallel` region of `slicewise_svdx`.

In both cases the crash persisted.

---

## Using AddressSanitizer to Debug Memory Issues

### Overview

AddressSanitizer (ASan) is a runtime memory error detector that can identify memory leaks,
invalid frees, heap buffer overflows, use-after-free, and similar issues. The pystarm build
system includes support for building with ASan.

### Building the ASan Target

Build pystarm with ASan instrumentation. This overwrites the production `.so` — running
`make clean` first is required if switching from a production build:

```bash
make clean
make BUILD_TYPE=asan
```

### System Setup

> **Note:** The commands in the ASan section below are run from the repo root
> (not from `paper/` like the section above).

Before running with ASan, the system environment must be configured correctly. A helper script
is provided:

```bash
source asan-run-prep.sh
```

### Running and Capturing ASan Output

Redirect both stdout and stderr to capture the full ASan report:

```bash
python paper/soccer.py -alg tsvdmi -k 50 -mtype eye > asan_soccer_eye.txt 2>&1
```

ASan output is written to stderr. The report will contain the error type, the full stack trace
at the point of the violation, and (for leaks) a summary of all unreleased allocations with
their allocation sites.

The ASan output from the crash run described in this document is saved in
`known_issues/asan_soccer_eye.txt` — refer to it as a concrete example of what ASan output
looks like for this type of error.
