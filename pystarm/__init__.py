"""pystarm: tensor algebra kernels backed by Intel MKL.

The two environment settings below must be in place before MKL initializes,
which is why they are here rather than in a shell script.

pystarm is compiled against MKL's 64-bit integer interface (-DMKL_ILP64) and
the GNU OpenMP threading layer. Other packages in a typical scientific Python
environment pull in MKL's dynamic dispatcher, libmkl_rt, which defaults to the
32-bit integer interface. When both are loaded into one process, whichever
resolves a symbol first wins, and pystarm's 64-bit integer arguments can be
read as 32-bit.

That failure is quiet and intermittent. It depends on import order, so the
same test passes on one run and fails on the next. Symptoms observed before
this was found: `Intel oneMKL ERROR: Parameter 9 was incorrect on entry to
cblas_ddgmm_batch` printed to stderr while execution continued with garbage,
three tests in the suite failing non-deterministically, and a multithreaded
crash recorded in known_issues/README.md that could never be reproduced
single-threaded. Forcing the interface layer removed all of them.

Both are set with setdefault, so an explicit choice by the caller or the
batch script still wins.

LIMITATION worth knowing: MKL reads these when it first initializes. If
something else in the process has already triggered that, for instance by
importing numpy or scipy and performing a linear algebra call, setting them
here is too late. Importing pystarm early, or exporting them in the shell,
is the reliable route. nersc-env-setup.sh and asan-run-prep.sh set them.
"""

import os as _os
import warnings as _warnings

# Deliberately NOT setting MKL_INTERFACE_LAYER here.
#
# The obvious fix for the clash described above is to force the 64-bit
# interface, and it does silence pystarm's symptoms. It also segfaults numpy:
# with a dynamically linked MKL the setting is process-global, so numpy's
# linear algebra, which expects the 32-bit interface, crashes in
# numpy.linalg.solve. Measured, reproducibly, on this environment.
#
# There is no environment variable that satisfies both. The fixes are
# structural: link MKL statically, as the default Makefile target does, so
# pystarm does not share the dynamic dispatcher; or build pystarm for the
# 32-bit interface; or use a numpy that does not link MKL at all.
_layer_was_set_by_caller = "MKL_INTERFACE_LAYER" in _os.environ

from .pystarm import *
from .alg import *


def _check_mkl_interface():
    """Warn when both MKL integer interfaces are loaded in this process.

    The setdefault above only helps when pystarm is imported before anything
    else initializes MKL. Importing numpy or scipy first and touching linear
    algebra is enough to lose that race, and then the mismatch is silent: MKL
    prints an argument complaint to stderr, if anything, and returns garbage.

    This check cannot repair the situation, and deliberately does not try.
    Forcing MKL_INTERFACE_LAYER=ILP64 silences pystarm's symptoms and then
    segfaults numpy, which expects the other interface from the same shared
    dispatcher. The check exists so the failure is visible rather than
    mysterious, and so the message can point at the structural fixes. Reading the process map is Linux-specific; anywhere else
    the check simply does nothing.
    """
    try:
        with open("/proc/self/maps") as fh:
            mapped = fh.read()
    except OSError:
        return

    if "libmkl_intel_lp64" in mapped and "libmkl_intel_ilp64" in mapped:
        # Only a caller-supplied setting is trustworthy here. Our own
        # setdefault proves nothing: if we had to apply it, something else
        # may already have initialized MKL with the other interface.
        if (not _layer_was_set_by_caller
                or _os.environ.get("MKL_INTERFACE_LAYER", "").upper() != "ILP64"):
            _warnings.warn(
                "Both MKL integer interfaces (lp64 and ilp64) are loaded in "
                "this process and MKL_INTERFACE_LAYER is not set to ILP64. "
                "pystarm is compiled for the 64-bit interface, so its integer "
                "arguments may be read as 32-bit, which produces wrong results "
                "intermittently rather than an error. "
                "Do NOT simply export MKL_INTERFACE_LAYER=ILP64: that fixes "
                "pystarm and segfaults numpy, which needs the other interface "
                "from the same dispatcher. Link MKL statically (the default "
                "Makefile target), or build pystarm for the 32-bit interface, "
                "or use a numpy that does not link MKL.",
                RuntimeWarning,
                stacklevel=2,
            )


_check_mkl_interface()
