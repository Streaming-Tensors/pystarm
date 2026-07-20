#!/usr/bin/env python3
"""
dload-ncep-data.py — Download NCEP Reanalysis data from NOAA PSL FTP

NCEP Reanalysis data is hosted at:
    https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html

Files are served over FTP at:
    ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis/{level}/{variable}.{year}.nc

Each file contains one year of a single variable at a given pressure level or
surface type. Files are in NetCDF format (.nc).

--------------------------------------------------------------------
LEVEL DIRECTORIES (--level)
--------------------------------------------------------------------
  pressure        Pressure-level data (multi-level, 17 standard pressure levels)
  surface         Surface or near-surface data (single level)
  surface_gauss   Surface data on Gaussian grid
  tropopause      Tropopause-level data
  other_gauss     Other Gaussian-grid variables

--------------------------------------------------------------------
COMMON VARIABLES (--variable)
--------------------------------------------------------------------
  Pressure-level variables (use with --level pressure):
    air       Air temperature
    hgt       Geopotential height
    omega     Vertical velocity (omega, Pa/s)
    rhum      Relative humidity
    shum      Specific humidity
    uwnd      U-wind (zonal wind component)
    vwnd      V-wind (meridional wind component)

  Surface variables (use with --level surface):
    air.sig995   Near-surface air temperature (sigma=0.995 level)
    slp          Sea level pressure
    uwnd.sig995  Near-surface U-wind
    vwnd.sig995  Near-surface V-wind
    rhum.sig995  Near-surface relative humidity
    pres.sfc     Surface pressure
    pr_wtr.eatm  Precipitable water (entire atmosphere)

  Tropopause variables (use with --level tropopause):
    air          Air temperature at tropopause
    hgt          Geopotential height at tropopause
    pres         Pressure at tropopause
    uwnd         U-wind at tropopause
    vwnd         V-wind at tropopause

NCEP Reanalysis covers 1948–present. Not all variables are available for all
years; check the PSL catalog if a year is missing.

--------------------------------------------------------------------
USAGE EXAMPLES
--------------------------------------------------------------------
  # Download pressure-level air temperature for 1980–1990 into ./data/
  # Files are saved to ./data/pressure/air.{year}.nc
  python dload-ncep-data.py --year-start 1980 --year-end 1990 \\
      --level pressure --variable air --outdir ./data

  # Download sea level pressure — saved to ./ncep/surface/slp.{year}.nc
  python dload-ncep-data.py --year-start 1948 --year-end 2024 \\
      --level surface --variable slp --outdir ./ncep

  # Use 4 threads instead of auto-detected core count
  python dload-ncep-data.py --year-start 2000 --year-end 2010 \\
      --level pressure --variable uwnd --outdir ./ncep --threads 4

  # Skip files that have already been downloaded (resume interrupted run)
  python dload-ncep-data.py --year-start 1948 --year-end 2024 \\
      --level pressure --variable hgt --outdir ./ncep --skip-existing
--------------------------------------------------------------------
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import concurrent.futures
import threading
from pathlib import Path

# Base FTP URL for NCEP Reanalysis data hosted at NOAA PSL
BASE_URL = "ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis"

# Lock used to serialise console output from multiple threads
_print_lock = threading.Lock()


def log(msg: str) -> None:
    """Thread-safe print to stdout."""
    with _print_lock:
        print(msg, flush=True)


def build_url(base_url: str, level: str, variable: str, year: int) -> str:
    """
    Construct the FTP URL for a single variable/year file.

    URL pattern:  {base_url}/{level}/{variable}.{year}.nc

    Parameters
    ----------
    base_url  : Root FTP URL (no trailing slash).
    level     : Subdirectory, e.g. 'pressure', 'surface'.
    variable  : Variable name, e.g. 'air', 'slp', 'uwnd.sig995'.
    year      : Four-digit year integer.

    Returns
    -------
    Full FTP URL string.
    """
    return f"{base_url}/{level}/{variable}.{year}.nc"


def download_year(
    base_url: str,
    level: str,
    variable: str,
    year: int,
    outdir: Path,
    skip_existing: bool,
) -> tuple[int, bool, str]:
    """
    Download the NetCDF file for one year of a given variable.

    Parameters
    ----------
    base_url      : Root FTP URL.
    level         : Level subdirectory (e.g. 'pressure').
    variable      : Variable name (e.g. 'air').
    year          : Year to download.
    outdir        : Local directory to save the file.
    skip_existing : If True, skip download when the destination file already
                    exists and is non-empty.

    Returns
    -------
    (year, success, message)
        year    : The year that was processed.
        success : True if the file is available locally after this call.
        message : Human-readable status string.
    """
    url = build_url(base_url, level, variable, year)
    filename = f"{variable}.{year}.nc"
    dest = outdir / filename

    # --- Skip if already downloaded ---
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return year, True, f"[SKIP]  {filename} (already exists)"

    # --- Attempt download ---
    try:
        tmp_dest = dest.with_suffix(".nc.part")
        urllib.request.urlretrieve(url, tmp_dest)
        # Rename only after a complete download so partial files are not kept
        tmp_dest.rename(dest)
        size_mb = dest.stat().st_size / (1024 ** 2)
        return year, True, f"[OK]    {filename}  ({size_mb:.1f} MB)"
    except urllib.error.URLError as exc:
        # Clean up partial file on failure
        if tmp_dest.exists():
            tmp_dest.unlink()
        return year, False, f"[FAIL]  {filename}  — {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        if tmp_dest.exists():
            tmp_dest.unlink()
        return year, False, f"[FAIL]  {filename}  — {exc}"


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    -------
    Parsed Namespace object with validated fields.
    """
    parser = argparse.ArgumentParser(
        prog="dload-ncep-data.py",
        description=(
            "Download NCEP Reanalysis NetCDF files from the NOAA PSL FTP server.\n"
            "One file per year is downloaded; files are named {variable}.{year}.nc.\n\n"
            "See the module docstring (top of this file) for a full list of levels,\n"
            "variables, and usage examples."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --year-start 1980 --year-end 1990 \\\n"
            "      --level pressure --variable air --outdir ./data\n"
            "  → saves to ./data/pressure/air.{year}.nc\n\n"
            "  %(prog)s --year-start 1948 --year-end 2024 \\\n"
            "      --level surface --variable slp --outdir ./ncep --skip-existing\n"
            "  → saves to ./ncep/surface/slp.{year}.nc\n"
        ),
    )

    parser.add_argument(
        "--year-start",
        type=int,
        required=True,
        metavar="YEAR",
        help="First year to download (inclusive). NCEP Reanalysis starts at 1948.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        required=True,
        metavar="YEAR",
        help="Last year to download (inclusive).",
    )
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        metavar="LEVEL",
        help=(
            "FTP subdirectory for the data level. "
            "Common values: pressure, surface, surface_gauss, tropopause, other_gauss."
        ),
    )
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        metavar="VAR",
        help=(
            "Variable name as it appears in the filename, e.g. 'air', 'hgt', "
            "'slp', 'uwnd.sig995'. See module docstring for full list."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        metavar="DIR",
        help=(
            "Base output directory. A subdirectory named after --level is created "
            "automatically inside it, e.g. --outdir ./data with --level pressure "
            "saves files to ./data/pressure/. Created if absent."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel download threads. "
            "Defaults to the number of logical CPU cores on this machine "
            f"(detected: {os.cpu_count()})."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip years whose output file already exists and is non-empty (resume mode).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        metavar="URL",
        help=f"Override the root FTP URL. Default: {BASE_URL}",
    )

    args = parser.parse_args()

    # --- Validate year range ---
    if args.year_start > args.year_end:
        parser.error(
            f"--year-start ({args.year_start}) must be ≤ --year-end ({args.year_end})"
        )
    if args.year_start < 1948:
        parser.error("NCEP Reanalysis starts at 1948; --year-start must be ≥ 1948.")

    # --- Validate thread count ---
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be a positive integer.")

    return args


def main() -> int:
    """
    Entry point: parse arguments, create output directory, and launch downloads.

    Returns
    -------
    0 on full success, 1 if any year failed to download.
    """
    args = parse_args()

    # Automatically nest files under {outdir}/{level}/ to mirror the FTP layout
    # and keep data from different levels separate when sharing a single base dir.
    base_outdir = Path(args.outdir)
    outdir = base_outdir / args.level
    outdir.mkdir(parents=True, exist_ok=True)

    years = list(range(args.year_start, args.year_end + 1))
    n_threads = args.threads if args.threads is not None else os.cpu_count()
    # Cap threads at the number of years — no point having idle threads
    n_threads = min(n_threads, len(years))

    print(
        f"\nNCEP Reanalysis downloader\n"
        f"  Variable : {args.variable}\n"
        f"  Level    : {args.level}\n"
        f"  Years    : {args.year_start}–{args.year_end}  ({len(years)} files)\n"
        f"  Base dir : {base_outdir.resolve()}\n"
        f"  Output   : {outdir.resolve()}\n"
        f"  Threads  : {n_threads}\n"
        f"  Base URL : {args.base_url}\n"
    )

    failures: list[int] = []

    # ThreadPoolExecutor: submit one task per year; each task runs in its own
    # thread up to n_threads concurrent threads.  Results are collected in
    # completion order via as_completed so progress prints promptly.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_year = {
            executor.submit(
                download_year,
                args.base_url,
                args.level,
                args.variable,
                year,
                outdir,
                args.skip_existing,
            ): year
            for year in years
        }

        for future in concurrent.futures.as_completed(future_to_year):
            year, success, message = future.result()
            log(message)
            if not success:
                failures.append(year)

    # --- Summary ---
    n_ok = len(years) - len(failures)
    print(f"\nDone: {n_ok}/{len(years)} files downloaded successfully.")
    if failures:
        failures.sort()
        print(f"Failed years: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
