"""
paired_stats.py
---------------
Consumes the JSON from multiseed_runner.py and produces:
  (a) Table II numbers (mean +/- std for every model, every metric).
  (b) A paired Wilcoxon signed-rank table across the GNN cluster (GCN, GAT,
      EvolveGCN, T-GAT) on AUC and F1 -- this is the table you paste into
      section 5.1 as the formal statistical-overlap evidence.

USAGE:
    !python paired_stats.py
       (expects /content/results/multiseed_results.json)

OUTPUT:
    /content/results/table_II.csv
    /content/results/wilcoxon_pairs.csv
    Prints both tables to stdout in markdown form so you can paste straight
    into a manuscript edit.
"""

import json, csv
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon

RES_PATH = Path('/content/results/multiseed_results.json')
OUT_DIR  = Path('/content/results')

METRICS  = ['auc', 'f1', 'precision', 'recall']
MODELS   = ['LR', 'RF', 'GCN', 'GAT', 'EvolveGCN', 'TGAT']
GNN_CLUSTER = ['GCN', 'GAT', 'EvolveGCN', 'TGAT']

def main():
    with open(RES_PATH) as f:
        blob = json.load(f)

    # Re-organise: per_model_metric[model][metric] = [values per seed]
    per_model_metric = {m: {k: [] for k in METRICS} for m in MODELS}
    for block in blob['results']:
        for m in MODELS:
            for k in METRICS:
                per_model_metric[m][k].append(block['models'][m][k])

    # ---- Table II ----------------------------------------------------------
    print('\n## Table II  (mean +/- std over {} seeds)\n'.format(len(blob['seeds'])))
    print('| Model | AUC-ROC | F1 (illicit) | Precision | Recall |')
    print('|---|---|---|---|---|')
    rows = []
    for m in MODELS:
        cells = []
        for k in METRICS:
            v = np.array(per_model_metric[m][k])
            cells.append(f'{v.mean():.3f} +/- {v.std(ddof=1):.3f}')
        rows.append([m] + cells)
        print('| ' + ' | '.join([m] + cells) + ' |')

    with open(OUT_DIR / 'table_II.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model'] + METRICS); w.writerows(rows)

    # ---- Wilcoxon signed-rank on GNN cluster -------------------------------
    print('\n## Paired Wilcoxon signed-rank, GNN cluster\n')
    print('| Pair | AUC p | AUC W | F1 p | F1 W |')
    print('|---|---|---|---|---|')
    wrows = []
    for a, b in combinations(GNN_CLUSTER, 2):
        for metric in ['auc', 'f1']:
            pass  # handled below to keep one row per pair
        a_auc = np.array(per_model_metric[a]['auc']); b_auc = np.array(per_model_metric[b]['auc'])
        a_f1  = np.array(per_model_metric[a]['f1']);  b_f1  = np.array(per_model_metric[b]['f1'])
        # zero-method='wilcox' drops zero-diff pairs (Wilcoxon's classical handling)
        try:
            ws_auc, p_auc = wilcoxon(a_auc, b_auc, zero_method='wilcox')
        except ValueError:  # all-zero differences -> identical -> trivially same
            ws_auc, p_auc = float('nan'), 1.0
        try:
            ws_f1,  p_f1  = wilcoxon(a_f1,  b_f1,  zero_method='wilcox')
        except ValueError:
            ws_f1, p_f1 = float('nan'), 1.0
        wrows.append([f'{a}-{b}', p_auc, ws_auc, p_f1, ws_f1])
        print(f'| {a}-{b} | {p_auc:.3f} | {ws_auc:.1f} | {p_f1:.3f} | {ws_f1:.1f} |')

    with open(OUT_DIR / 'wilcoxon_pairs.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['pair', 'auc_p', 'auc_W', 'f1_p', 'f1_W']); w.writerows(wrows)

    # Headline interpretation for the manuscript
    print('\n## Interpretation\n')
    sig_auc = sum(1 for r in wrows if r[1] < 0.05)
    sig_f1  = sum(1 for r in wrows if r[3] < 0.05)
    total = len(wrows)
    print(f'{sig_auc}/{total} GNN pairs separable at p<0.05 on AUC; '
          f'{sig_f1}/{total} separable on F1.')
    if sig_auc == 0 and sig_f1 == 0:
        print('PASTE INTO SECTION 5.1: '
              '"A paired Wilcoxon signed-rank test across {} seeds did not '
              'separate any pair of GNN models at the 5% level on either '
              'AUC-ROC or illicit-class F1."'.format(len(blob['seeds'])))

if __name__ == '__main__':
    main()
