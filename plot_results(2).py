"""
plot_results.py
---------------
Generates the publication figures from the JSON output of multiseed_runner.py
and fidelity_audit_extended.py.

Outputs (all 300 dpi PNG, ready to drop into the manuscript):
  /content/results/fig4_auc_f1.png        (Figure 4 panels a + b)
  /content/results/fig4c_per_seed_scatter.png  (Figure 4 panel c, the new one)
  /content/results/fig5_fidelity.png      (Figure 5 panels a + b)

USAGE:
    !python plot_results.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RES_DIR = Path('/content/results')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'savefig.dpi': 300,
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

MODEL_ORDER = ['LR', 'RF', 'GCN', 'GAT', 'EvolveGCN', 'TGAT']
MODEL_LABEL = {'LR':'LR', 'RF':'RF', 'GCN':'GCN', 'GAT':'GAT',
               'EvolveGCN':'EvolveGCN', 'TGAT':'T-GAT'}
GNN_MODELS  = ['GCN', 'GAT', 'EvolveGCN', 'TGAT']


def fig4_ab(blob):
    """Bar chart of AUC and F1 with 1-sigma error bars. RF filled, GNNs hatched
    to make the cluster visually obvious. Includes a shaded band over the GNN
    cluster on each panel."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), constrained_layout=True)
    for ax, metric, label in [(axes[0], 'auc', 'AUC-ROC'),
                              (axes[1], 'f1',  'Illicit-class F1')]:
        means, stds = [], []
        for m in MODEL_ORDER:
            v = np.array([r['models'][m][metric] for r in blob['results']])
            means.append(v.mean()); stds.append(v.std(ddof=1))
        means, stds = np.array(means), np.array(stds)
        colors = ['#bbbbbb' if m != 'RF' else '#222244' for m in MODEL_ORDER]
        bars = ax.bar(range(len(MODEL_ORDER)), means, yerr=stds, capsize=3,
                      color=colors, edgecolor='black', linewidth=0.5)
        # shaded GNN band
        gnn_idx = [MODEL_ORDER.index(m) for m in GNN_MODELS]
        gnn_means = means[gnn_idx]; gnn_stds = stds[gnn_idx]
        band_lo = (gnn_means - gnn_stds).min()
        band_hi = (gnn_means + gnn_stds).max()
        ax.axhspan(band_lo, band_hi, alpha=0.08, color='steelblue', zorder=0)
        ax.set_xticks(range(len(MODEL_ORDER)))
        # Steeper rotation + right-anchored alignment fixes the
        # EvolveGCN / T-GAT overlap that 20-degree rotation produced.
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER],
                           rotation=35, ha='right')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0 if metric == 'auc' else max(0.9, (means+stds).max()*1.1))
        # value labels
        for i, (mu, sd) in enumerate(zip(means, stds)):
            ax.text(i, mu + sd + 0.01, f'{mu:.3f}', ha='center', fontsize=8)
    axes[0].set_title('(a) AUC-ROC across seeds')
    axes[1].set_title('(b) F1-score across seeds')
    out = RES_DIR / 'fig4_auc_f1.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {out}')


def fig4c_scatter(blob):
    """Per-seed AUC scatter for GNN models -- the new panel that shows the
    within-model spread is comparable to the between-model gap."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    xs = np.arange(len(GNN_MODELS))
    for i, m in enumerate(GNN_MODELS):
        vals = [r['models'][m]['auc'] for r in blob['results']]
        ax.scatter([i] * len(vals), vals, s=40, edgecolor='black',
                   facecolor='steelblue', alpha=0.75, zorder=3)
        ax.hlines(np.mean(vals), i - 0.18, i + 0.18, color='black', linewidth=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL[m] for m in GNN_MODELS])
    ax.set_ylabel('AUC-ROC (per seed)')
    ax.set_title('(c) Per-seed AUC scatter across GNN models')
    out = RES_DIR / 'fig4c_per_seed_scatter.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {out}')


def fig5_fidelity(fblob):
    """Two panels: per-node drop, and paired full-vs-pruned predictions.
    Plots both GNNExplainer and PGExplainer if both are present. Because the
    two explainers can produce identical Fidelity values on Elliptic (both
    degenerate to non-discriminative masks under a feature-driven decision
    boundary), we draw GNNExplainer as larger filled circles and PGExplainer
    as smaller red crosses on top, so superimposed points show both markers
    rather than hiding one behind the other."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), constrained_layout=True)

    def gnn_kwargs():
        # filled circle, large, semi-transparent so the cross on top is visible
        return dict(marker='o', s=46, c='#1f77b4',
                    edgecolors='black', linewidths=0.5, alpha=0.55, zorder=2)

    def pg_kwargs():
        # cross marker -- unfilled, so we pass the line color via `c`.
        # `linewidths` controls the cross stroke thickness.
        return dict(marker='x', s=32, c='#d62728',
                    linewidths=1.4, alpha=1.0, zorder=3)

    mean_line = {'GNNExplainer': '#1f77b4', 'PGExplainer': '#d62728'}

    # Panel (a): per-node drop, sorted
    ax = axes[0]
    for key, kw_fn in (('GNNExplainer', gnn_kwargs), ('PGExplainer', pg_kwargs)):
        if key not in fblob: continue
        drops = sorted([r['drop'] for r in fblob[key]['per_node']])
        xs = list(range(1, len(drops) + 1))
        ax.scatter(xs, drops, label=key, **kw_fn())
        ax.axhline(np.mean(drops), color=mean_line[key], linestyle='--',
                   linewidth=1, alpha=0.65,
                   label=f'{key} mean = {np.mean(drops):.3f}')
    ax.set_xlabel('Flagged illicit node (rank by drop)')
    ax.set_ylabel(r'Fidelity drop $\Delta = \hat{y}_{\mathrm{full}} - \hat{y}_{\mathrm{pruned}}$')
    ax.set_title('(a) Per-node Fidelity drop')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

    # Panel (b): paired predictions
    ax = axes[1]
    for key, kw_fn in (('GNNExplainer', gnn_kwargs), ('PGExplainer', pg_kwargs)):
        if key not in fblob: continue
        rows = fblob[key]['per_node']
        full = [r['full'] for r in rows]; pruned = [r['pruned'] for r in rows]
        ax.scatter(full, pruned, label=key, **kw_fn())
    lim = [0.92, 1.0]
    ax.plot(lim, lim, 'k--', linewidth=0.8, alpha=0.5, label='y = x (no drop)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'Fraud probability -- full graph')
    ax.set_ylabel(r'Fraud probability -- top-$k$ edges removed')
    ax.set_title('(b) Predictions barely change after edge removal')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

    out = RES_DIR / 'fig5_fidelity.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {out}')


def main():
    multi_path = RES_DIR / 'multiseed_results.json'
    fid_path   = RES_DIR / 'fidelity_audit_extended.json'
    if multi_path.exists():
        with open(multi_path) as f: blob = json.load(f)
        fig4_ab(blob); fig4c_scatter(blob)
    else:
        print('Skipped Figure 4: run multiseed_runner.py first.')
    if fid_path.exists():
        with open(fid_path) as f: fblob = json.load(f)
        fig5_fidelity(fblob)
    else:
        print('Skipped Figure 5: run fidelity_audit_extended.py first.')

if __name__ == '__main__':
    main()
