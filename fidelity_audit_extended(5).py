"""
fidelity_audit_extended.py
--------------------------
Extends the Fidelity audit of section 5.3 from 20 nodes -> 100 nodes, and
adds PGExplainer alongside GNNExplainer. This is the experiment that strengthens
the "near-zero Fidelity" claim from a single-explainer observation to a
dataset-level observation.

INPUTS:
  Assumes the trained T-GAT checkpoint sits at /content/results/tgat_seed42.pt
  (i.e. the seed-42 model from multiseed_runner.py). If a different seed is
  preferred, change SEED below; results are insensitive to seed choice in
  our preliminary runs.

OUTPUTS:
  /content/results/fidelity_audit_extended.json
  /content/results/fidelity_audit_extended.csv

USAGE (on Colab):
    !python fidelity_audit_extended.py

The summary stats printed at the end are what gets pasted into the [TODO] block
in section 5.3 of the manuscript.
"""

import os, json, math, csv, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# PyTorch 2.6+ compatibility shim. See multiseed_runner.py for the rationale.
# ---------------------------------------------------------------------------
def _enable_pyg_load():
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return
    safe = []
    try:
        from torch_geometric.data.data import Data; safe.append(Data)
    except ImportError:
        pass
    for mod, names in [
        ("torch_geometric.data.data", ["DataEdgeAttr", "DataTensorAttr"]),
        ("torch_geometric.data.storage",
         ["GlobalStorage", "NodeStorage", "EdgeStorage", "BaseStorage"]),
    ]:
        try:
            m = __import__(mod, fromlist=names)
            for n in names:
                cls = getattr(m, n, None)
                if cls is not None:
                    safe.append(cls)
        except ImportError:
            pass
    if safe:
        try:
            torch.serialization.add_safe_globals(safe)
        except Exception:
            pass
    _orig_load = torch.load
    def _patched(f, map_location=None, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(f, map_location=map_location, **kw)
    torch.load = _patched

_enable_pyg_load()
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.nn import GATv2Conv

# import T-GAT definition from the multiseed file -- keeps the models identical
# (alternatively paste the class here if you prefer a fully standalone script)
try:
    from multiseed_runner import TGAT, sinusoidal_time_encoding, TIME_DIM, HIDDEN, DROPOUT
except ImportError:
    raise SystemExit('Place multiseed_runner.py next to this script before running.')

SEED          = 42
K_EDGES       = 30          # top-k edges removed per node (matches section 5.3)
N_NODES       = 100         # extended from 20
PROB_THRESHOLD = 0.997      # high-confidence flagged illicit on test set
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR       = Path('/content/results'); OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def load_data_and_model():
    ds = EllipticBitcoinDataset(root='/content/elliptic')
    data = ds[0]
    data.train_mask = data.train_mask & (data.y != 2)
    data.test_mask  = data.test_mask  & (data.y != 2)
    data = data.to(DEVICE)

    in_dim, out_dim = data.x.size(1), 2
    model = TGAT(in_dim, HIDDEN // 4, out_dim).to(DEVICE)
    ckpt = '/content/results/tgat_seed42.pt'
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f'Loaded {ckpt}')
    else:
        raise SystemExit(f'Missing {ckpt}. Save the seed-{SEED} T-GAT weights '
                         'at the end of multiseed_runner.py first.')
    model.eval()

    # t_index for T-GAT forward pass
    t_path = '/content/elliptic/t_index.pt'
    t_index = (torch.load(t_path).to(DEVICE) if os.path.exists(t_path)
               else torch.full((data.num_nodes,), 25, device=DEVICE))
    return data, model, t_index


def predict(model, data, t_index):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, t_index)
    return F.softmax(logits, dim=1)[:, 1]


def pick_flagged_nodes(probs, mask, n=N_NODES):
    """Top-n highest-confidence illicit predictions on the test set."""
    cand = torch.where(mask & (probs > PROB_THRESHOLD))[0]
    if cand.numel() < n:
        # fall back: top-n by probability on test set
        scored = probs.masked_fill(~mask, -1.0)
        return torch.topk(scored, n).indices.tolist()
    ordered = cand[torch.argsort(-probs[cand])][:n]
    return ordered.tolist()


# -------- model wrapper that the explainers expect --------------------------
class TGATWrapper(torch.nn.Module):
    """Explainer expects model(x, edge_index, **kwargs). We bake t_index in."""
    def __init__(self, inner, t_index):
        super().__init__(); self.inner = inner; self.t_index = t_index
    def forward(self, x, edge_index):
        return self.inner(x, edge_index, self.t_index)


def fidelity_for_node(wrapped, data, node, mask, k=K_EDGES):
    """Edge mask values -> top-k -> remove those edges -> Delta(probability)."""
    # full-graph prediction
    with torch.no_grad():
        full = F.softmax(wrapped(data.x, data.edge_index), dim=1)[node, 1].item()
    # top-k edges by mask value
    top = torch.topk(mask, k=min(k, mask.numel())).indices
    keep = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=DEVICE)
    keep[top] = False
    pruned_ei = data.edge_index[:, keep]
    with torch.no_grad():
        pruned = F.softmax(wrapped(data.x, pruned_ei), dim=1)[node, 1].item()
    return full, pruned, full - pruned


def run_gnnexplainer(wrapped, data, flagged):
    """Runs GNNExplainer and incrementally writes per-node results to disk so
    a downstream crash never costs us this pass again."""
    cache_path = OUT_DIR / 'fidelity_gnnexplainer.json'
    rows = []
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if (cached.get('config', {}).get('k') == K_EDGES
                and len(cached.get('per_node', [])) == len(flagged)
                and {r['node'] for r in cached['per_node']} == set(flagged)):
                print(f'  GNNExplainer cache hit ({len(cached["per_node"])} nodes) -> skip')
                return cached['per_node']
        except Exception as e:
            print(f'  GNNExplainer cache exists but unreadable ({e}); recomputing')

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=100, lr=0.01),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification',
                          task_level='node', return_type='log_probs'),
    )
    for i, n in enumerate(flagged, 1):
        out = explainer(data.x, data.edge_index, index=n)
        full, pruned, drop = fidelity_for_node(wrapped, data, n, out.edge_mask)
        rows.append({'node': int(n), 'full': full, 'pruned': pruned, 'drop': drop})
        if i % 10 == 0:
            print(f'  GNNExplainer  {i}/{len(flagged)}')
            # Save incrementally every 10 nodes -- cheap, makes the pass crash-safe.
            with open(cache_path, 'w') as f:
                json.dump({
                    'config': {'k': K_EDGES, 'n_nodes': len(flagged)},
                    'per_node': rows,
                }, f)
    # Final flush
    with open(cache_path, 'w') as f:
        json.dump({
            'config': {'k': K_EDGES, 'n_nodes': len(flagged)},
            'per_node': rows,
        }, f)
    return rows


def run_pgexplainer(wrapped, data, flagged):
    """Trains PGExplainer's parameter network, then evaluates Fidelity on the
    same flagged set.

    PyG >= 2.6 requires the algorithm to be 'connected' to a model config
    before train() can be called. The Explainer wrapper does this in its
    own __init__, so the documented PyG idiom is:

        1. Build the algorithm (PGExplainer)
        2. Wrap it in Explainer(...)            <- this calls algorithm.connect()
        3. Train via explainer.algorithm.train(...)
        4. Run inference via explainer(...)

    My earlier code did step 3 before step 2 and silently failed. This
    version follows the documented order.
    """
    pg = PGExplainer(epochs=30, lr=0.003).to(DEVICE)

    # Step 2 -- creating the Explainer wires algorithm.connect(...) internally.
    explainer = Explainer(
        model=wrapped,
        algorithm=pg,
        explanation_type='phenomenon',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification',
                          task_level='node', return_type='log_probs'),
    )
    # Defensive: in some PyG builds the algorithm's submodules are lazily
    # constructed on first .train() call and end up on CPU. Move them once
    # more after Explainer has done its setup.
    pg.to(DEVICE)

    print('  Training PGExplainer parameter network ...')
    train_set = flagged[:20]
    for epoch in range(30):
        pg.to(DEVICE)
        for n in train_set:
            explainer.algorithm.train(
                epoch, wrapped, data.x, data.edge_index,
                target=data.y, index=n)
        if (epoch + 1) % 5 == 0:
            print(f'  PGExplainer training epoch {epoch+1}/30')

    cache_path = OUT_DIR / 'fidelity_pgexplainer.json'
    rows = []
    for i, n in enumerate(flagged, 1):
        out = explainer(data.x, data.edge_index, index=n, target=data.y)
        full, pruned, drop = fidelity_for_node(wrapped, data, n, out.edge_mask)
        rows.append({'node': int(n), 'full': full, 'pruned': pruned, 'drop': drop})
        if i % 10 == 0:
            print(f'  PGExplainer   {i}/{len(flagged)}')
            with open(cache_path, 'w') as f:
                json.dump({
                    'config': {'k': K_EDGES, 'n_nodes': len(flagged)},
                    'per_node': rows,
                }, f)
    with open(cache_path, 'w') as f:
        json.dump({
            'config': {'k': K_EDGES, 'n_nodes': len(flagged)},
            'per_node': rows,
        }, f)
    return rows


def summary(name, rows):
    drops = np.array([r['drop'] for r in rows])
    return {
        'explainer': name,
        'n_nodes'  : len(rows),
        'mean_drop': float(drops.mean()),
        'std_drop' : float(drops.std(ddof=1)),
        'median_drop': float(np.median(drops)),
        'min_drop' : float(drops.min()),
        'max_drop' : float(drops.max()),
    }


def main():
    set_seed(SEED)
    data, model, t_index = load_data_and_model()
    wrapped = TGATWrapper(model, t_index).to(DEVICE).eval()

    probs = predict(model, data, t_index)
    flagged = pick_flagged_nodes(probs, data.test_mask, N_NODES)
    print(f'Auditing {len(flagged)} flagged nodes (predicted p > {PROB_THRESHOLD}).')

    # ---- GNNExplainer (cached, crash-safe) -------------------------------
    gnnex_rows = run_gnnexplainer(wrapped, data, flagged)
    gnnex_summary = summary('GNNExplainer', gnnex_rows)

    # ---- PGExplainer (may fail on some PyG versions) ---------------------
    pg_summary, pg_rows, pg_error = None, [], None
    try:
        pg_rows = run_pgexplainer(wrapped, data, flagged)
        pg_summary = summary('PGExplainer', pg_rows)
    except Exception as e:
        import traceback
        pg_error = f'{type(e).__name__}: {e}'
        print('\n!!! PGExplainer failed; GNNExplainer results are still saved.')
        print(f'    Error: {pg_error}')
        traceback.print_exc()

    # ---- Combined output -------------------------------------------------
    blob = {
        'config': {'seed': SEED, 'k': K_EDGES, 'n_nodes': N_NODES,
                   'p_thresh': PROB_THRESHOLD},
        'GNNExplainer': {'summary': gnnex_summary, 'per_node': gnnex_rows},
    }
    if pg_summary is not None:
        blob['PGExplainer'] = {'summary': pg_summary, 'per_node': pg_rows}
    else:
        blob['PGExplainer_error'] = pg_error

    with open(OUT_DIR / 'fidelity_audit_extended.json', 'w') as f:
        json.dump(blob, f, indent=2)

    with open(OUT_DIR / 'fidelity_audit_extended.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['explainer', 'node', 'full', 'pruned', 'drop'])
        for r in gnnex_rows:
            w.writerow(['GNNExplainer', r['node'], r['full'], r['pruned'], r['drop']])
        for r in pg_rows:
            w.writerow(['PGExplainer',  r['node'], r['full'], r['pruned'], r['drop']])

    print('\n=== Summary ===')
    print(f"GNNExplainer    mean={gnnex_summary['mean_drop']:.4f} "
          f"std={gnnex_summary['std_drop']:.4f} "
          f"median={gnnex_summary['median_drop']:.4f}  "
          f"(n={gnnex_summary['n_nodes']}, k={K_EDGES})")
    if pg_summary is not None:
        print(f"PGExplainer     mean={pg_summary['mean_drop']:.4f} "
              f"std={pg_summary['std_drop']:.4f} "
              f"median={pg_summary['median_drop']:.4f}  "
              f"(n={pg_summary['n_nodes']}, k={K_EDGES})")
    else:
        print('PGExplainer     SKIPPED (see error above)')
    print(f'\nWrote {OUT_DIR}/fidelity_audit_extended.{{json,csv}}')


if __name__ == '__main__':
    main()
