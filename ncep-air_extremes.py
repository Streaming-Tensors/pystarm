"""
ncep-air_extremes.py — Compare EOF and tsvdmii reconstruction quality at
extreme temperature events.

For each grid point (lat, lon) at a given pressure level, extreme events are
defined as time steps in the top 0.5% (hot) and bottom 0.5% (cold) of the
original temperature distribution — 1% combined. For each extreme time step
the pointwise relative error of each reconstruction is computed. Mean and max
errors are aggregated into global maps (73 x 144) for each method.

Output: a single PDF with 4 global maps (2x2 layout):
    EOF     — mean relative error at extremes
    EOF     — max relative error at extremes
    tsvdmii — mean relative error at extremes
    tsvdmii — max relative error at extremes

Usage:
    python ncep-air_extremes.py \
        --data-dir   /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
        --year-start 1948 --year-end 1957 \
        --tsvdmii-dir /pscratch/sd/t/taufique/pystarm/extremes/ncep-air_tsvdmii_0.01_dct_0123_64 \
        --eof-dir     /pscratch/sd/t/taufique/pystarm/extremes/ncep-air_eof_0.01_1000_64 \
        --level-hpa  850 \
        --outdir     /pscratch/sd/t/taufique/pystarm/extremes
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from experiments import read_ncep_air

# NCEP standard pressure levels (hPa) in the order they appear in the data
# after transposing to (lat, lon, level, time). Index 0 = 1000 hPa, index 2 = 850 hPa.
NCEP_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250,
                   200, 150, 100, 70, 50, 30, 20, 10]


def level_index(level_hpa):
    """Return the index of a pressure level in the NCEP levels list."""
    if level_hpa not in NCEP_LEVELS_HPA:
        raise ValueError(f"{level_hpa} hPa not in NCEP levels: {NCEP_LEVELS_HPA}")
    return NCEP_LEVELS_HPA.index(level_hpa)


def compute_extreme_errors(orig, reconst, p_low=0.5, p_high=99.5):
    """
    For each grid point (lat, lon), identify extreme time steps and compute
    pointwise relative errors against the reconstruction.

    Parameters
    ----------
    orig    : np.ndarray, shape (nlat, nlon, ntime) — original temperature slice
    reconst : np.ndarray, shape (nlat, nlon, ntime) — reconstructed temperature slice
    p_low   : lower percentile threshold (cold extremes)
    p_high  : upper percentile threshold (hot extremes)

    Returns
    -------
    mean_err : np.ndarray, shape (nlat, nlon) — mean relative error at extremes
    max_err  : np.ndarray, shape (nlat, nlon) — max relative error at extremes
    """
    nlat, nlon, ntime = orig.shape
    mean_err   = np.zeros((nlat, nlon), dtype=np.float64)
    max_err    = np.zeros((nlat, nlon), dtype=np.float64)
    median_err = np.zeros((nlat, nlon), dtype=np.float64)

    for i in range(nlat):
        for j in range(nlon):
            ts_orig    = orig[i, j, :]       # (ntime,) time series at this grid point
            ts_reconst = reconst[i, j, :]

            # Identify extreme time steps: top 0.5% and bottom 0.5%
            lo = np.percentile(ts_orig, p_low)
            hi = np.percentile(ts_orig, p_high)
            extreme_mask = (ts_orig <= lo) | (ts_orig >= hi)
            extreme_idx  = np.where(extreme_mask)[0]

            if len(extreme_idx) == 0:
                continue

            # Pointwise relative error at each extreme time step
            orig_vals    = ts_orig[extreme_idx]
            reconst_vals = ts_reconst[extreme_idx]
            errors = np.abs(orig_vals - reconst_vals) / np.abs(orig_vals)

            mean_err[i, j]   = np.mean(errors)
            max_err[i, j]    = np.max(errors)
            median_err[i, j] = np.median(errors)

    return mean_err, max_err, median_err


def plot_maps(maps, titles, lats, lons, suptitle, outpath):
    """
    Plot 2 global maps stacked vertically, sized for a single ACM column (~3.33in wide).
    Colorbar is placed on the right side, shared across both panels.

    Parameters
    ----------
    maps    : list of 2 np.ndarray, each shape (nlat, nlon)
    titles  : list of 2 strings
    lats    : 1D array of latitudes
    lons    : 1D array of longitudes
    suptitle: overall figure title
    outpath : output PDF path
    """
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 1, figsize=(3.33, 4.0),
                             subplot_kw={"projection": proj})

    # Use a shared color scale across both maps for direct comparison
    vmin = min(m.min() for m in maps)
    vmax = max(m.max() for m in maps)

    for ax, data, title in zip(axes, maps, titles):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        im = ax.pcolormesh(lons, lats, data,
                           transform=proj,
                           cmap='YlOrRd',
                           vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=6)

    fig.suptitle(suptitle, fontsize=6)

    # Colorbar on the right, shared across both panels
    fig.subplots_adjust(right=0.82, hspace=0.15)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Relative error')
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('Relative error', fontsize=5)

    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot to {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare EOF and tsvdmii reconstruction quality at extreme events."
    )
    parser.add_argument("--data-dir",     type=str, required=True,
                        help="Directory containing air.{year}.nc files")
    parser.add_argument("--year-start",   type=int, default=1948,
                        help="First year to load inclusive (default: 1948)")
    parser.add_argument("--year-end",     type=int, default=1957,
                        help="Last year to load inclusive (default: 1957)")
    parser.add_argument("--tsvdmii-dir",  type=str, required=True,
                        help="Run directory from ncep-air_tsvdmii.py (contains reconstruction.npy)")
    parser.add_argument("--eof-dir",      type=str, required=True,
                        help="Run directory from ncep-air_eof.py (contains reconstruction.npy)")
    parser.add_argument("--level-hpa",    type=int, default=850,
                        help="Pressure level in hPa for extreme detection (default: 850)")
    parser.add_argument("--outdir",       type=str, required=True,
                        help="Output directory for the PDF plot")
    parser.add_argument("--metric",       type=str, default="median",
                        choices=["mean", "max", "median"],
                        help="Error metric to plot (default: median)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lev_idx  = level_index(args.level_hpa)
    out_pdf  = os.path.join("plots", f"extremes_{args.level_hpa}hpa_{args.metric}.pdf")

    # --- Load original data ---
    print(f"Loading original NCEP air data: {args.year_start}–{args.year_end}")
    t0  = time.perf_counter()
    arr = read_ncep_air(args.data_dir, "air", args.year_start, args.year_end)
    print(f"Load time: {time.perf_counter() - t0:.1f}s  shape: {arr.shape}")

    # --- Load reconstructions ---
    print("\nLoading tsvdmii reconstruction...")
    t0 = time.perf_counter()
    arr_tsvdmii = np.load(os.path.join(args.tsvdmii_dir, "reconstruction.npy"))
    print(f"Load time: {time.perf_counter() - t0:.1f}s  shape: {arr_tsvdmii.shape}")

    print("Loading EOF reconstruction...")
    t0 = time.perf_counter()
    arr_eof = np.load(os.path.join(args.eof_dir, "reconstruction.npy"))
    print(f"Load time: {time.perf_counter() - t0:.1f}s  shape: {arr_eof.shape}")

    # --- Extract pressure level slice (lat, lon, time) ---
    orig    = arr[:, :, lev_idx, :]         # (73, 144, time)
    reconst_tsvdmii = arr_tsvdmii[:, :, lev_idx, :]
    reconst_eof     = arr_eof[:,    :, lev_idx, :]

    # --- Compute extreme errors (skip if already saved) ---
    tsvdmii_mean_file = os.path.join(args.tsvdmii_dir, f"mean_err_{args.level_hpa}hpa.npy")
    tsvdmii_max_file  = os.path.join(args.tsvdmii_dir, f"max_err_{args.level_hpa}hpa.npy")
    eof_mean_file     = os.path.join(args.eof_dir,     f"mean_err_{args.level_hpa}hpa.npy")
    eof_max_file      = os.path.join(args.eof_dir,     f"max_err_{args.level_hpa}hpa.npy")

    tsvdmii_median_file = os.path.join(args.tsvdmii_dir, f"median_err_{args.level_hpa}hpa.npy")
    eof_median_file     = os.path.join(args.eof_dir,     f"median_err_{args.level_hpa}hpa.npy")

    if all(os.path.exists(f) for f in [tsvdmii_mean_file, tsvdmii_max_file, tsvdmii_median_file,
                                        eof_mean_file, eof_max_file, eof_median_file]):
        print(f"\nError maps found — skipping computation, loading saved arrays.")
        mean_tsvdmii   = np.load(tsvdmii_mean_file)
        max_tsvdmii    = np.load(tsvdmii_max_file)
        median_tsvdmii = np.load(tsvdmii_median_file)
        mean_eof       = np.load(eof_mean_file)
        max_eof        = np.load(eof_max_file)
        median_eof     = np.load(eof_median_file)
    else:
        print(f"\nComputing extreme errors at {args.level_hpa} hPa ...")
        print("  tsvdmii ...")
        t0 = time.perf_counter()
        mean_tsvdmii, max_tsvdmii, median_tsvdmii = compute_extreme_errors(orig, reconst_tsvdmii)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

        print("  EOF ...")
        t0 = time.perf_counter()
        mean_eof, max_eof, median_eof = compute_extreme_errors(orig, reconst_eof)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

        np.save(tsvdmii_mean_file,   mean_tsvdmii)
        np.save(tsvdmii_max_file,    max_tsvdmii)
        np.save(tsvdmii_median_file, median_tsvdmii)
        np.save(eof_mean_file,       mean_eof)
        np.save(eof_max_file,        max_eof)
        np.save(eof_median_file,     median_eof)
        print(f"Saved error maps to {args.tsvdmii_dir} and {args.eof_dir}")

    # --- Lat/lon arrays for plotting ---
    # NCEP lat: 90N -> 90S, 2.5 degree spacing (73 points)
    # NCEP lon: 0 -> 357.5E, 2.5 degree spacing (144 points)
    lats = np.linspace(90,  -90, arr.shape[0])
    lons = np.linspace(0,  357.5, arr.shape[1])

    # --- Select metric maps ---
    metric_maps = {
        "mean":   (mean_eof,   mean_tsvdmii),
        "max":    (max_eof,    max_tsvdmii),
        "median": (median_eof, median_tsvdmii),
    }
    eof_map, tsvdmii_map = metric_maps[args.metric]

    # --- Plot ---
    print("\nPlotting...")
    maps   = [eof_map, tsvdmii_map]
    titles = [
        f"EOF — {args.metric} relative error at extremes",
        f"tsvdmii — {args.metric} relative error at extremes",
    ]
    suptitle = (f"Reconstruction error at extreme events ({args.level_hpa} hPa)  "
                f"[{args.year_start}–{args.year_end}]")
    plot_maps(maps, titles, lats, lons, suptitle, out_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()
