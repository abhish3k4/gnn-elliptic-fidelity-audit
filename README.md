# gnn-elliptic-fidelity-audit
Multi-seed reproducibility audit of GNN architectures and a Fidelity audit of two post-hoc subgraph explainers (GNNExplainer, PGExplainer) on the Elliptic Bitcoin benchmark.

gnn-elliptic-fidelity-audit/
├── README.md
├── LICENSE
├── .gitignore
├── scripts/
│   ├── multiseed_runner.py
│   ├── paired_stats.py
│   ├── fidelity_audit_extended.py
│   └── plot_results.py
├── results/
│   ├── multiseed_results.json
│   ├── multiseed_probs.json
│   ├── fidelity_audit_extended.json
│   ├── fidelity_audit_extended.csv
│   ├── wilcoxon_pairs.csv
│   └── table_II.csv
├── checkpoints/
│   └── tgat_seed42.pt
└── figures/
    ├── fig4_auc_f1.png
    ├── fig4c_per_seed_scatter.png
    └── fig5_fidelity.png
