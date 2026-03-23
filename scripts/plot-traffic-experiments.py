"""
plot-traffic-experiments.py

Generates relative error vs compression ratio plots for traffic experiments.

Three sets of PDFs saved to scripts/plots/:
  compare_alg_{mtype}_{perm}.pdf   — tsvdmi vs tsvdmii, one per (mtype, perm)
  compare_mtype_{alg}_{perm}.pdf   — dct vs eye vs hosvd, one per (alg, perm)
  compare_perm_{mtype}_{alg}.pdf   — perm 0123 vs 0321, one per (mtype, alg)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, 'experiments.csv')
PLOTS_DIR  = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df['complete'] == True].copy()

for col in ['compression_ratio', 'relative_err', 'k', 'tol']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Plot helpers ──────────────────────────────────────────────────────────────
ALG_STYLES  = {'tsvdmi':  {'color': 'tab:blue',   'marker': 'o'},
               'tsvdmii': {'color': 'tab:orange',  'marker': 's'}}

MTYPE_STYLES = {'dct':   {'color': 'tab:blue',   'marker': 'o'},
                'eye':   {'color': 'tab:green',  'marker': 's'},
                'hosvd': {'color': 'tab:red',    'marker': '^'}}

PERM_STYLES  = {'(0, 1, 2, 3)': {'color': 'tab:blue',  'marker': 'o'},
                '(0, 3, 2, 1)': {'color': 'tab:orange', 'marker': 's'}}

PERM_LABELS  = {'(0, 1, 2, 3)': 'perm=0123',
                '(0, 3, 2, 1)': 'perm=0321'}


def annotate_points(ax, grp, x_col, y_col):
    """Annotate each point with its k or tol value."""
    for _, row in grp.iterrows():
        if pd.notna(row.get('k')):
            label = f"k={int(row['k'])}"
        elif pd.notna(row.get('tol')):
            label = f"tol={row['tol']}"
        else:
            continue
        ax.annotate(label, (row[x_col], row[y_col]),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)


def style_ax(ax, title, xlabel='Relative Error', ylabel='Compression Ratio'):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ── 1. Compare alg (tsvdmi vs tsvdmii) for each (mtype, perm) ────────────────
print('\n--- Compare alg ---')
for mtype in df['mtype'].dropna().unique():
    for perm in df['perm_mode'].dropna().unique():
        subset = df[(df['mtype'] == mtype) & (df['perm_mode'] == perm)]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        for alg, style in ALG_STYLES.items():
            grp = subset[subset['alg'] == alg].sort_values('relative_err')
            if grp.empty:
                continue
            ax.plot(grp['relative_err'], grp['compression_ratio'],
                    label=alg, **style)
            annotate_points(ax, grp, 'relative_err', 'compression_ratio')

        perm_label = PERM_LABELS.get(perm, perm)
        style_ax(ax, f'tsvdmi vs tsvdmii  |  mtype={mtype}  |  {perm_label}')
        save_fig(fig, f'compare_alg_{mtype}_{perm_label.replace("perm=", "")}.pdf')


# ── 2. Compare mtype (dct vs eye vs hosvd) for each (alg, perm) ──────────────
print('\n--- Compare mtype ---')
for alg in df['alg'].dropna().unique():
    for perm in df['perm_mode'].dropna().unique():
        subset = df[(df['alg'] == alg) & (df['perm_mode'] == perm)]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        for mtype, style in MTYPE_STYLES.items():
            grp = subset[subset['mtype'] == mtype].sort_values('relative_err')
            if grp.empty:
                continue
            ax.plot(grp['relative_err'], grp['compression_ratio'],
                    label=f'M={mtype}', **style)
            annotate_points(ax, grp, 'relative_err', 'compression_ratio')

        perm_label = PERM_LABELS.get(perm, perm)
        style_ax(ax, f'mtype comparison  |  {alg}  |  {perm_label}')
        save_fig(fig, f'compare_mtype_{alg}_{perm_label.replace("perm=", "")}.pdf')


# ── 3. Compare perm (0123 vs 0321) for each (mtype, alg) ─────────────────────
print('\n--- Compare perm ---')
for mtype in df['mtype'].dropna().unique():
    for alg in df['alg'].dropna().unique():
        subset = df[(df['mtype'] == mtype) & (df['alg'] == alg)]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        for perm, style in PERM_STYLES.items():
            grp = subset[subset['perm_mode'] == perm].sort_values('relative_err')
            if grp.empty:
                continue
            ax.plot(grp['relative_err'], grp['compression_ratio'],
                    label=PERM_LABELS.get(perm, perm), **style)
            annotate_points(ax, grp, 'relative_err', 'compression_ratio')

        style_ax(ax, f'perm mode comparison  |  mtype={mtype}  |  {alg}')
        save_fig(fig, f'compare_perm_{mtype}_{alg}.pdf')

print('\nDone. All plots saved to:', PLOTS_DIR)

# ── Merge all PDFs into one ───────────────────────────────────────────────────
from pypdf import PdfWriter

merged_path = os.path.join(PLOTS_DIR, 'traffic-experiments-all.pdf')
writer = PdfWriter()

groups = ['compare_alg', 'compare_mtype', 'compare_perm']
pdf_files = []
for group in groups:
    group_files = sorted(
        f for f in os.listdir(PLOTS_DIR)
        if f.startswith(group) and f.endswith('.pdf')
    )
    pdf_files.extend(group_files)

for fname in pdf_files:
    writer.append(os.path.join(PLOTS_DIR, fname))

with open(merged_path, 'wb') as f:
    writer.write(f)

print(f'Merged PDF saved to: {merged_path}')
