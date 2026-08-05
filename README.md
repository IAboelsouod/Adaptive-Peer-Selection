# Adaptive Peer Selection for Hierarchical Federated Learning

Flower simulation comparing **flat FedAvg**, **static hierarchical FL**, and **adaptive peer selection** under controlled non-IID data. Clients are assigned to logical edge aggregators via cosine similarity of model-update deltas and graph community detection, with periodic global sync following Abad et al.'s Algorithm 2.

## Problem statement

Federated learning under non-IID client data suffers when every client trains against a single global model. Hierarchical federated learning (HFL) mitigates this by grouping clients under edge aggregators that run intra-cluster FedAvg before a less-frequent global merge. The assignment of clients to edge aggregators is critical: a poor grouping wastes communication and hurts accuracy.

This project asks: **can we recover latent client structure from update similarity and adapt edge-aggregator assignment at runtime**, improving both task accuracy and cluster recovery versus flat FedAvg and static HFL?

We simulate three systems on Fashion-MNIST with Dirichlet label skew and planted latent groups:

| ID | Strategy | Description |
|----|----------|-------------|
| **A** | `flat` | Standard FedAvg — one global model every round |
| **B** | `hier_static` | HFL with fixed random client→edge assignment |
| **C** | `adaptive` | HFL with adaptive assignment from delta cosine similarity → peer graph → Louvain communities |

## Related work

**Abad et al.** (ICASSP 2020) introduce hierarchical FL across heterogeneous cellular networks: clients cluster under edge aggregators, intra-cluster FedAvg runs every round, and a global model is formed every *H* rounds (Algorithm 2). Their client-to-aggregator assignment is **static and distance-based**. We keep the same *H*-periodic sync schedule and logical three-tier loop but replace static assignment with a **data-similarity-driven, adaptive** one under **non-IID** label skew.

**Rana et al.** frame aggregator placement and user-equipment (UE) assignment as a **dynamic placement problem** in hierarchical and decentralised FL (HierFedML). Our adaptive peer selection is a concrete instance of that problem: at each global-sync boundary we re-solve “which clients share an edge aggregator?” from observed update geometry rather than a fixed partition.

**FedAvg** (McMahan et al., AISTATS 2017) is the flat baseline and the within-cluster aggregation rule. Clustered-FL work (e.g. Sattler et al.) clusters clients by update cosine similarity; we combine that idea with Abad et al.'s hierarchical sync cadence.

## Method

1. **Logical hierarchy inside one Flower Strategy** — no separate edge-server processes. The custom `Strategy` groups fit results by edge aggregator, aggregates within clusters, and merges globally on sync rounds.
2. **Delta similarity** — per client \( \Delta_i = w_i - w_{\text{global}} \) (last-layer by default); pairwise cosine similarity matrix.
3. **Peer graph** — NetworkX graph, edges sparsified by threshold and top-*k*; weight = similarity.
4. **Community detection** — Louvain (default) maps communities → edge aggregators; optional hysteresis stabilises group IDs across rounds.
5. **Warm-up** — first `peers.warmup_rounds` use random grouping (early deltas are noisy).
6. ***H*-periodic global sync** — every `hierarchy.global_sync_every` rounds, cluster models merge to a global model (Abad et al. Algorithm 2). Between syncs, clients may receive their cluster model (`model_distribution: cluster`).

See [DESIGN.md](DESIGN.md) for full terminology and config knobs.

## Repository map

```
├── config.yaml              # Default experiment configuration
├── DESIGN.md                # Architecture and terminology (source of truth)
├── Makefile / run.sh        # install, test, baseline, adaptive, sweep, figures
├── requirements.txt
├── src/
│   ├── config.py            # Typed YAML loader
│   ├── run.py               # Single-run entrypoint
│   ├── client.py            # Flower NumPyClient
│   ├── data/partition.py    # Dirichlet non-IID + latent groups
│   ├── models/cnn.py        # CNN models for fmnist/mnist/cifar10
│   ├── strategies/          # flat_fedavg, hier_static, adaptive_peers
│   ├── peers/               # similarity matrix, graph, community detection
│   ├── metrics/logging.py   # Per-round CSV logger
│   └── viz/                 # Peer-graph snapshots and accuracy curves
├── scripts/
│   ├── experiments.py       # Multi-run sweep driver
│   ├── aggregate_results.py # Master table + comm-cost metrics
│   └── generate_figures.py  # Headline figure pipeline
├── tests/                   # Unit tests (partition, selection, aggregation)
└── results/                 # CSV logs and figures (gitignored)
```

## Quick start

```bash
make install          # create .venv and install dependencies
make test             # unit tests + tiny smoke simulation
make figures          # headline plots (short matched A/B/C run)
```

Equivalent via shell wrapper:

```bash
chmod +x run.sh
./run.sh install
./run.sh test
./run.sh figures
```

Python **3.10+** required (Flower 1.28). Python 3.11+ recommended. First run downloads Fashion-MNIST automatically.

## Makefile targets

| Target | What it does |
|--------|----------------|
| `install` | Create `.venv`, `pip install -r requirements.txt` |
| `test` | `pytest tests/` then smoke FL runs (3 rounds, 8 clients) |
| `baseline` | Single flat FedAvg run (`baseline_flat`) |
| `adaptive` | Single adaptive run (`baseline_adaptive`, α=0.1, H=2) |
| `sweep` | Reduced grid (α=0.1, H∈{1,2}, 2 seeds) + `master_table.csv` |
| `figures` | Matched A/B/C short run + all headline figures |

Override run length, e.g. `make figures NUM_ROUNDS_FIGURES=10 NUM_CLIENTS_FIGURES=20`.

## Reproducing figures and tables

### Headline figures (`make figures`)

Produces:

| Output | Description |
|--------|-------------|
| `results/accuracy_abc_overlay.png` | Global accuracy vs round — A/B/C overlay |
| `results/cluster_quality_adaptive.png` | ARI & NMI vs round (adaptive) |
| `results/summary.csv` | Convergence table (rounds-to-target, final acc/ARI) |
| `results/summary_table.png` | Rendered summary table |
| `results/summary_table.md` | Markdown summary |
| `results/peer_graphs/peer_graph_round_*.png` | Community-coloured peer graphs (fixed layout) |
| `results/peer_graph_evolution.gif` | Peer-graph snapshots stitched over training |

Manual equivalent:

```bash
python scripts/generate_figures.py --num-rounds 8 --num-clients 12
```

Publication-quality (longer) run:

```bash
python scripts/generate_figures.py --num-rounds 10 --num-clients 50 --run-prefix phase6
```

### Sweep and master table (`make sweep`)

```bash
make sweep                                    # reduced sweep (default)
python scripts/experiments.py --reduced --num-rounds 10
python scripts/aggregate_results.py
```

Full grid (long):

```bash
python scripts/experiments.py --num-rounds 10
python scripts/aggregate_results.py
```

Outputs `results/master_table.csv` and `results/master_table.md` with mean ± std final accuracy, final ARI, rounds-to-target (90% of flat final acc), communication-cost proxy, and `accuracy_by_H` trade-off column.

### Single runs

```bash
python -m src.run --strategy flat --run-name my_flat --num-rounds 10
python -m src.run --strategy hier_static --alpha 0.1 --global-sync-every 2 --run-name my_static
python -m src.run --strategy adaptive --alpha 0.1 --global-sync-every 2 --run-name my_adaptive
```

## Results (reduced sweep, α=0.1, cluster distribution)

From `results/master_table.csv` (2 seeds, 10 rounds, 50 clients):

- **Adaptive (H=1)** reaches **0.79** mean final accuracy with **ARI ≈ 0.36**, beating flat (**0.74**) at **~5× lower** communication cost (cheap client↔edge every round; expensive edge↔global only on sync rounds).
- **Accuracy vs *H***: raising *H* from 1→2 cuts expensive global transfers and yields **~7×** cost improvement for adaptive, but accuracy drops to **0.74** — the expected accuracy–communication trade-off from Abad et al.
- **Static HFL** matches adaptive on accuracy at *H*=1 but **ARI ≈ 0** (random fixed groups do not recover latent structure).
- **Peer-graph figures** (`results/peer_graph_evolution.gif`) show communities forming after warm-up with **stable node layout and hysteresis-preserved colours** across rounds.
- **Overlay curve** (`results/accuracy_abc_overlay.png`) and **cluster-quality plot** (`results/cluster_quality_adaptive.png`) visualise the accuracy gap and rising ARI/NMI during adaptive training.

## References

1. M. S. H. Abad, E. Ozfatura, D. Gündüz, O. Ercetin. *Hierarchical Federated Learning Across Heterogeneous Cellular Networks.* ICASSP 2020, pp. 8866–8870.
2. O. Rana, T. Spyridopoulos, N. Hudson, M. Baughman, K. Chard, I. Foster, A. Khan. *Hierarchical and Decentralised Federated Learning.*
3. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017 (FedAvg).

## License

Academic / coursework use. See course submission guidelines for your institution.
