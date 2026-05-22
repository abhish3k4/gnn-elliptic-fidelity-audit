# GNN Elliptic Fidelity Audit

Multi-seed reproducibility audit of Graph Neural Network architectures and a
quantitative Fidelity audit of two post-hoc subgraph explainers
(GNNExplainer, PGExplainer) on the Elliptic Bitcoin benchmark.

This repository contains the complete pipeline behind the paper:

> **Graph Neural Networks for Bitcoin Fraud Detection: A Multi-Seed Empirical
> Study with a Fidelity Audit of Post-Hoc Explanations.**
> Abhishek Singh. Department of Computer Science and Engineering, Kamla Nehru
> Institute of Technology, Sultanpur, Uttar Pradesh, India.
> [Submission in progress; DOI will be inserted at proof stage.]

## Headline findings

1. Across ten random seeds, the four GNN architectures evaluated here
   (GCN, GAT, EvolveGCN, T-GAT) cluster within a narrow AUC band of
   0.828 – 0.845. A paired Wilcoxon signed-rank test fails to separate the
   proposed T-GAT from a plain GAT on either AUC (p = 0.28) or F1 (p = 0.23).
2. Random Forest substantially outperforms every GNN on raw classification
   (AUC = 0.926 &plusmn; 0.005, F1 = 0.822 &plusmn; 0.002), with the gap to the
   best GNN roughly seven times larger than the largest gap within the GNN
   cluster.
3. Both GNNExplainer and PGExplainer yield numerically identical Fidelity
   scores on every flagged node (mean drop 0.006 &plusmn; 0.020,
   median 0.000, 71 / 100 nodes show zero drop). The post-hoc "critical
   sub-graph" is therefore not causally responsible for the prediction on
   Elliptic for either explainer.

## What is in this repository

```
.
├── scripts/
│   ├── multiseed_runner.py         # 10-seed evaluation of all 6 baselines
│   ├── paired_stats.py             # Table II and paired Wilcoxon tests
│   ├── fidelity_audit_extended.py  # 100-node Fidelity audit, both explainers
│   └── plot_results.py             # Figures 4(a)(b)(c) and 5(a)(b)
├── results/
│   ├── multiseed_results.json      # per-seed AUC / F1 / P / R for all models
│   ├── multiseed_probs.json        # raw per-node probabilities (for re-eval)
│   ├── fidelity_audit_extended.json
│   ├── fidelity_audit_extended.csv
│   ├── wilcoxon_pairs.csv          # pairwise p-values, GNN cluster
│   └── table_II.csv                # mean +/- std table
├── checkpoints/
│   └── tgat_seed42.pt              # weights used in the Fidelity audit
├── figures/
│   ├── fig4_auc_f1.png             # main AUC + F1 bar chart
│   ├── fig4c_per_seed_scatter.png  # per-seed AUC scatter, GNN cluster
│   └── fig5_fidelity.png           # 100-node Fidelity audit
└── README.md (this file)
```

## How to reproduce the results

The full pipeline runs on the free tier of Google Colab (NVIDIA T4 GPU, 15 GB
VRAM) in under three hours of compute.

### 1. Environment

```bash
pip install torch-geometric==2.5.3 captum==0.7.0 scikit-learn==1.5 scipy matplotlib
```

PyTorch 2.6 changed the default of `torch.load(weights_only=...)` from
`False` to `True`, which breaks loading of the cached PyG dataset files.
The scripts in this repository contain a compatibility shim that allowlists
the relevant PyG classes; no user action required.

### 2. Reproduce Table II and the Wilcoxon tests

```bash
python scripts/multiseed_runner.py
python scripts/paired_stats.py
```

Wall-clock: roughly 15 minutes on a T4. Outputs `results/multiseed_results.json`
and `results/multiseed_probs.json`, plus prints Table II and the pairwise
Wilcoxon table.

### 3. Reproduce the Fidelity audit

```bash
python scripts/fidelity_audit_extended.py
```

Wall-clock: roughly 60 – 70 minutes. Loads `checkpoints/tgat_seed42.pt`,
selects the 100 highest-confidence flagged illicit transactions on the test
set, and runs both GNNExplainer and PGExplainer with k = 30 edges removed per
node. Outputs `results/fidelity_audit_extended.json` and `.csv`.

The script caches GNNExplainer's per-node results to disk every 10 nodes, so
if the PGExplainer phase fails (it has had unstable APIs across PyG releases),
the GNNExplainer results are preserved and the script exits cleanly with a
`PGExplainer SKIPPED` message.

### 4. Regenerate the figures

```bash
python scripts/plot_results.py
```

Wall-clock: a few seconds. Writes three 300 dpi PNGs to `figures/`.

## Random seeds

The ten seeds used throughout the paper are `42, 43, 44, 45, 46, 47, 48, 49,
50, 51`. PyTorch, NumPy, and the CUDA back-end are all seeded explicitly per
run. Deterministic kernels are enabled where supported.

## Hardware used in the paper

| Item | Specification |
|---|---|
| GPU | NVIDIA T4, 15 GB VRAM (Google Colab free tier) |
| CPU | Intel Xeon (Colab default) |
| RAM | 13 GB system memory |
| OS | Ubuntu 22.04 LTS (Colab default) |
| Python | 3.12 |
| PyTorch Geometric | 2.5.3 |
| Captum | 0.7.0 |

## Dataset

The Elliptic Bitcoin dataset is distributed by Elliptic Enterprises Ltd. and
hosted on Kaggle at
[https://www.kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
PyTorch Geometric downloads and processes the dataset automatically on first
use. No copy is redistributed in this repository.

The dataset contains approximately 203,000 anonymised Bitcoin transactions
arranged over 49 time steps with 166 anonymised features per node, with a
chronological 1 – 34 / 35 – 49 train / test split.

## Known caveats

1. Every edge in the released Elliptic graph connects two transactions in the
   same discrete time step, so the inter-step temporal mechanisms in the T-GAT
   architecture reduce to the identity transformation on this benchmark. The
   results in this repository are therefore informative about the GNN family
   and the explainer family on a feature-driven, heterophilous dataset; they
   may not transfer to account-based chains with cross-step edges such as
   Ethereum.
2. The Fidelity audit uses k = 30 as the number of removed edges per node.
   Larger k values produce broadly similar results in our preliminary checks
   but a comprehensive sweep is outside the scope of the present paper.

## Citation

If you use this repository or build on these findings, please cite:

```
@article{singh2026gnnelliptic,
  title   = {Graph Neural Networks for Bitcoin Fraud Detection: A Multi-Seed
             Empirical Study with a Fidelity Audit of Post-Hoc Explanations},
  author  = {Singh, Abhishek},
  journal = {[TODO: journal name once accepted]},
  year    = {2026}
}
```

## License

MIT &mdash; see `LICENSE`.

## Contact

Abhishek Singh
Department of Computer Science and Engineering
Kamla Nehru Institute of Technology, Sultanpur, Uttar Pradesh, India
[abhishek.2410201@knit.ac.in](mailto:abhishek.2410201@knit.ac.in)
