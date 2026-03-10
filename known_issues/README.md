# Known Issues

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

---

## Using AddressSanitizer to Debug Memory Issues

### Overview

AddressSanitizer (ASan) is a runtime memory error detector that can identify memory leaks,
invalid frees, heap buffer overflows, use-after-free, and similar issues. The pystarm build
system includes support for building with ASan.

### Building the ASan Target

The `Makefile` contains an `asan` target that compiles pystarm with ASan instrumentation and
produces a separately named `.so` file (e.g., `pystarm_asan.cpython-311-x86_64-linux-gnu.so`)
so it does not conflict with the normal build:

```bash
make asan
```

### System Setup

Before running with ASan, the system environment must be configured correctly. A helper script
is provided:

```bash
source scripts/asan-run-prep.sh
```

### Loading the ASan Build in Python

Because the ASan build has a different filename, pystarm cannot be loaded with a plain
`import pystarm`. Instead, load it manually using `importlib`. The top of `soccer.py` contains
the necessary code — **uncomment** those lines and **comment out** the regular `import pystarm`
line:

```python
import importlib.util, sys
_spec = importlib.util.spec_from_file_location(
        "pystarm",
        "/path/to/pystarm_asan.cpython-311-x86_64-linux-gnu.so"
)
pystarm = importlib.util.module_from_spec(_spec)
sys.modules["pystarm"] = pystarm
_spec.loader.exec_module(pystarm)

# import pystarm   <-- comment this out
```

This also ensures that `alg.py` (which does `import pystarm`) picks up the ASan build
automatically, since the module is registered in `sys.modules` before `alg.py` is imported.

### Running and Capturing ASan Output

Redirect both stdout and stderr to capture the full ASan report:

```bash
python soccer.py -alg tsvdmi -k 50 -mtype eye > asan_soccer_eye.txt 2>&1
```

ASan output is written to stderr. The report will contain the error type, the full stack trace
at the point of the violation, and (for leaks) a summary of all unreleased allocations with
their allocation sites.

The ASan output from the crash run described in this document is saved in
`known_issues/asan_soccer_eye.txt` — refer to it as a concrete example of what ASan output
looks like for this type of error.
