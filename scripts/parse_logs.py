import os
import re
import csv
import argparse


def find(pattern, text):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None


def findall_sum(pattern, text):
    """Find all matches of a float pattern and return their sum, or None if no matches."""
    matches = re.findall(pattern, text, re.MULTILINE)
    return sum(float(x) for x in matches) if matches else None


def parse_filename(filename):
    """Parse experiment metadata from log filename.

    Expected format: {dname}_{alg}_{k_or_tol}_{mtype}_{perm_mode}_{omp_threads}
    Example: traffic_tsvdmii_0.1_eye_0123_64
    """
    parts = filename.split('_')
    if len(parts) < 6:
        return {}
    return {
        'dname_f':       parts[0],
        'alg_f':         parts[1],
        'k_or_tol_f':    parts[2],
        'mtype_f':       parts[3],
        'perm_mode_f':   parts[4],
        'omp_threads_f': parts[5],
    }


def parse_content(content):
    """Parse experiment results from log file content."""
    row = {}

    # --- Parameters printed by experiments.py ---
    row['alg']          = find(r'^alg\s+:\s+(\S+)', content)
    row['mtype']        = find(r'^mtype\s+:\s+(\S+)', content)
    row['k']            = find(r'^k\s+:\s+(\S+)', content)
    row['tol']          = find(r'^tol\s+:\s+(\S+)', content)
    row['dname']        = find(r'^dname\s+:\s+(\S+)', content)
    row['dfile']        = find(r'^dfile\s+:\s+(\S+)', content)
    row['perm_mode']    = find(r'^perm_mode\s*:\s+(.+)', content)
    row['tensor_shape'] = find(r'^Tensor shape:\s+(.+)', content)

    # --- Timings from experiments.py ---
    row['time_transform_matrix'] = find(r'^Time to generate transformation matrices:\s+([\d.e+\-]+)', content)
    row['time_data_to_pystarm']  = find(r'^Time to convert to pystarm tensor:\s+([\d.e+\-]+)', content)

    # --- Compressed representation size ---
    row['compression_ratio'] = find(r'^Compression ratio:\s+([\d.e+\-]+)', content)

    # --- Split content into compress and reconstruct sections ---
    # All [tsvdm_X_compress] lines come before [tsvdm_X_reconstruct] lines
    reconstruct_start   = re.search(r'^\[tsvdm_(?:I|II)_reconstruct\]', content, re.MULTILINE)
    compress_section    = content[:reconstruct_start.start()] if reconstruct_start else content
    reconstruct_section = content[reconstruct_start.start():] if reconstruct_start else ''

    # --- Compress timings from alg.py ---
    row['time_compress_ttm_total'] = findall_sum(
        r'^\[tsvdm_(?:I|II)_compress\] Time for TTM on mode \d+ :\s+([\d.e+\-]+)', compress_section)

    # tsvdmi compress
    row['time_slicewise_svd']     = find(r'^\[tsvdm_I_compress\] Time for slicewise SVD:\s+([\d.e+\-]+)', compress_section)

    # tsvdmii compress
    row['time_slicewise_svdvals'] = find(r'^\[tsvdm_II_compress\] Time for slicewise_svdvals:\s+([\d.e+\-]+)', compress_section)
    row['time_thresholds']        = find(r'^\[tsvdm_II_compress\] Time for thresholds:\s+([\d.e+\-]+)', compress_section)
    row['time_slicewise_svdks']   = find(r'^\[tsvdm_II_compress\] Time for slicewise_svdks:\s+([\d.e+\-]+)', compress_section)

    # --- Reconstruct timings from alg.py ---
    row['time_reconstruct_matmul']    = find(r'^\[tsvdm_(?:I|II)_reconstruct\] Time to reconstruct A_hat:\s+([\d.e+\-]+)', reconstruct_section)
    row['time_reconstruct_ttm_total'] = findall_sum(
        r'^\[tsvdm_(?:I|II)_reconstruct\] Time for TTM on mode \d+ :\s+([\d.e+\-]+)', reconstruct_section)

    # --- Error metrics ---
    row['absolute_err']       = find(r'^Absolute err:\s+([\d.e+\-]+)', content)
    row['relative_err']       = find(r'^Relative err:\s+([\d.e+\-]+)', content)
    row['norm_original']      = find(r'^Norm of original tensor:\s+([\d.e+\-]+)', content)
    row['norm_reconstructed'] = find(r'^Norm of reconstructed tensor:\s+([\d.e+\-]+)', content)

    row['complete'] = all(row.get(f) is not None for f in ['relative_err', 'absolute_err', 'compression_ratio'])

    return row


def parse_logfile(filepath):
    filename = os.path.basename(filepath)
    with open(filepath, 'r') as f:
        content = f.read()

    row = {'filename': filename}
    row.update(parse_filename(filename))
    row.update(parse_content(content))
    return row


COLUMNS = [
    'filename', 'complete',
    # From filename
    'dname_f', 'alg_f', 'k_or_tol_f', 'mtype_f', 'perm_mode_f', 'omp_threads_f',
    # Parameters from content
    'alg', 'mtype', 'k', 'tol', 'dname', 'dfile', 'perm_mode',
    # Tensor info
    'tensor_shape',
    # experiments.py timings
    'time_transform_matrix', 'time_data_to_pystarm',
    # Compressed size
    'compression_ratio',
    # Compress timings (alg.py)
    'time_compress_ttm_total',
    'time_slicewise_svd',           # tsvdmi only
    'time_slicewise_svdvals',       # tsvdmii only
    'time_thresholds',              # tsvdmii only
    'time_slicewise_svdks',         # tsvdmii only
    # Reconstruct timings (alg.py)
    'time_reconstruct_matmul',
    'time_reconstruct_ttm_total',
    # Error metrics
    'absolute_err', 'relative_err', 'norm_original', 'norm_reconstructed',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir",  type=str, help="Directory containing log files")
    parser.add_argument("outfile", type=str, help="Output CSV file path")
    args = parser.parse_args()

    logfiles = [f for f in os.listdir(args.logdir) if os.path.isfile(os.path.join(args.logdir, f))]
    rows = []
    for fname in sorted(logfiles):
        fpath = os.path.join(args.logdir, fname)
        rows.append(parse_logfile(fpath))
        print(f"Parsed: {fname}")

    with open(args.outfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.outfile}")
