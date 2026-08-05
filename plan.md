# Adaptive Peer Selection for Hierarchical Federated Learning — Cursor Build Plan

A phase-by-phase prompt plan to build the project with Cursor's agent. Each phase has a **goal**, a **copy-paste prompt**, the **files** it should touch, and an **acceptance check** so you can verify before moving on. Work one phase at a time — do not paste the whole thing at once.

---

## 0. What we're building (the one-paragraph spec)

A Flower-based FL **simulation** with three logical tiers — clients → edge aggregators → global server — where the **assignment of clients to edge aggregators is recomputed adaptively** (at each global sync) from the **cosine similarity of client model-update deltas**, replacing the static, distance-based assignment of Abad et al. (ICASSP 2020) with a data-driven one under non-IID data. Similar clients become *peers* sharing an edge aggregator; each edge group runs intra-group FedAvg; edge models are then aggregated into a global model. We compare three systems: **(A) flat FedAvg baseline**, **(B) hierarchical FL with static/random groups**, **(C) adaptive peer selection (ours)** — under controllable non-IID data, and we score both **task accuracy** and **cluster recovery quality** against a known latent group structure.

### Locked design decisions
- **Framework:** PyTorch + Flower (`flwr`) simulation. Pin `flwr` to a fixed 1.x version in `requirements.txt`.
- **Hierarchy lives inside a custom `Strategy`.** No separate edge-server processes. `aggregate_fit` receives all client results, groups them, aggregates within group (edge), then across groups (global).
- **Similarity signal:** per-client **update delta** `Δ_i = w_i^(t) − w_global^(t-1)`, flattened. Default to the **classifier / last-layer delta** to cut dimensionality and sharpen label-skew signal; keep full-delta as a config switch.
- **Peer selection:** build a weighted graph (NetworkX), nodes = clients, edge weight = cosine similarity, sparsify by threshold or top-k, run **community detection** (greedy modularity / Louvain). Each community = one edge aggregator. Number of groups is **emergent**, not fixed — that's the "adaptive" part. Keep a fixed-K spectral/agglomerative variant as a config option for ablation.
- **Warm-up:** first `warmup_rounds` use flat/random grouping (early deltas are noise). Re-cluster every `recluster_every` rounds with optional hysteresis to avoid thrashing.
- **Model distribution choice (decide early, expose as config):** clients receive either the **global** model (pure HFL) or their **cluster** model (personalized HFL). Run both; personalized usually wins under strong non-IID.
- **Global-sync cadence `H` (from Abad et al., Algorithm 2):** edge groups aggregate intra-cluster every round, but the global server averages cluster models only every `H` rounds, then redistributes. Between syncs, clients train on and receive their **cluster** model. Higher `H` = fewer global syncs = cheaper backhaul but more inter-cluster drift — this is the central HFL knob and the source of the accuracy/communication trade-off you report. Adaptive re-clustering happens at sync boundaries, when a fresh global reference exists to compute deltas against.

### Defaults (all in `config.yaml`, nothing hardcoded)
- Dataset: FMNIST (then MNIST, then CIFAR-10). Clients: 50. Rounds: 50. Local epochs: 2.
- Non-IID: Dirichlet `alpha` ∈ {0.1, 0.5, 1.0} **plus** a latent-group injection (e.g. 3 groups with disjoint-ish label support) so ground-truth cluster labels exist for ARI/NMI.
- Seeds fixed everywhere (torch, numpy, python, Flower).

---

## 1. Related work & positioning (use this for the report's intro / related-work)

The project sits between two anchor papers and extends them in a specific, defensible way.

**Abad et al., "Hierarchical Federated Learning across Heterogeneous Cellular Networks" (ICASSP 2020)** is the operational template. Their Algorithm 2 is the canonical 3-tier loop we implement: clients run local updates, a small-cell base station (SBS) aggregates within its cluster every round, and a macro base station (MBS) averages across clusters every `H` rounds before redistributing. Two of their choices are exactly what we change — client-to-cluster assignment is **static and distance-based** (geography picks your aggregator), and their experiments are **IID**, with non-IID explicitly named as future work.

**Rana et al., "Hierarchical and Decentralised Federated Learning"** frames the open problem we target: it treats *aggregation-node assignment* and *UE/client assignment* (its HierFedML reference) as a hard, ideally **dynamic** placement problem, and stresses the generalisation-vs-personalisation trade-off that non-IID locality forces.

**Contribution, stated precisely:** replace distance-based static assignment with **data-similarity-based adaptive peer selection** — cosine similarity of model-update deltas → community detection → client→aggregator assignment, recomputed at each global sync — evaluated under **non-IID** data. This is simultaneously Abad et al.'s named future work (non-IID HFL) and a data-driven instance of Rana et al.'s dynamic UE-assignment problem, solved with a learning signal rather than a wireless/geographic one.

**Method lineage to cite:** FedAvg (McMahan et al.) for aggregation; the intra-cluster→global `H`-periodic structure from Abad et al.; FedProx / SCAFFOLD as optional local-side correction under heterogeneity; FedPer / APFL / pFedMe for the cluster-model (personalised) distribution variant; and similarity-of-updates clustering in the spirit of Sattler et al.'s clustered FL (cited within Abad et al.) for the peer-selection mechanism itself.

**Scope note:** Abad et al.'s headline result is *communication-latency* reduction via an explicit wireless channel model. We deliberately drop the channel/resource-allocation modelling (out of scope for a data-driven simulation) and keep a **communication-cost proxy** instead — but we frame results in their terms: intra-cluster traffic is cheap, client→global traffic is expensive, and higher `H` means fewer global syncs.

---

## How to drive Cursor

- One phase = one focused agent run. Review the diff, run the acceptance check, commit, then next phase.
- Keep `DESIGN.md` (Phase 0) in context for every prompt — reference it by name so the agent stays consistent.
- After each phase, ask Cursor to run the relevant script/test and **paste the actual output back**, not just claim success.
- mininet is overkill here; NetworkX covers the visualization requirement. Skip mininet unless your supervisor explicitly wants real network emulation.

### `.cursorrules` (create this first, before Phase 0)
```
- Project: Adaptive Peer Selection for Hierarchical Federated Learning (FL simulation).
- Stack: Python 3.11, PyTorch, Flower (flwr) in simulation mode, scikit-learn, NetworkX, pandas, matplotlib. No TensorFlow. No mininet.
- The 3-tier hierarchy is LOGICAL, implemented inside a custom flwr Strategy. Do NOT spawn separate edge-server processes.
- All experiment parameters come from config.yaml via a typed config object. Never hardcode hyperparameters in logic.
- Set all random seeds for reproducibility. Log every round's metrics to a pandas DataFrame -> CSV under results/.
- Prefer small, typed, testable functions. Add docstrings. Keep aggregation math in pure functions that take/return numpy arrays so they can be unit-tested without Flower.
- Before writing new code, read DESIGN.md and match its terminology (peer, edge aggregator, community, delta).
```

### Target repo structure
```
fl-adaptive-peers/
├── config.yaml
├── requirements.txt
├── DESIGN.md
├── src/
│   ├── config.py            # typed config loader
│   ├── data/partition.py    # non-IID + latent-group partitioning
│   ├── models/cnn.py        # model defs + param (de)serialization helpers
│   ├── client.py            # Flower NumPyClient
│   ├── strategies/
│   │   ├── flat_fedavg.py    # baseline A
│   │   ├── hier_static.py    # baseline B (fixed groups)
│   │   └── adaptive_peers.py # system C (ours)
│   ├── peers/
│   │   ├── similarity.py     # delta extraction + cosine matrix
│   │   └── selection.py      # NetworkX graph + community detection
│   ├── metrics/logging.py    # pandas round logger
│   ├── viz/
│   │   ├── curves.py         # accuracy/loss/ARI plots
│   │   └── graph.py          # NetworkX peer-graph snapshots
│   └── run.py               # entrypoint: pick strategy via config
├── results/                 # csv logs + figures (gitignored)
└── tests/
```

---

## Phase 0 — Scaffolding, config, reproducibility

**Goal:** repo skeleton, dependency pinning, typed config, seed control, DESIGN.md.

**Prompt:**
```
Read .cursorrules. Create the repo structure shown below and scaffold the foundations only — no FL logic yet.

1. requirements.txt pinning: flwr[simulation], torch, torchvision, scikit-learn, networkx, python-louvain (community), pandas, matplotlib, pyyaml, pytest. Pin flwr to a specific recent 1.x version and note it.
2. config.yaml with sections: dataset, federation (num_clients, num_rounds, local_epochs, fraction_fit), hierarchy (global_sync_every: H, num_groups for the static baseline), data (alpha, num_latent_groups, partition_scheme), model (name), peers (similarity_source: last_layer|full, threshold, top_k, community_method, recluster_every, warmup_rounds, hysteresis), training (lr, batch_size, optimizer, local_optimizer: fedavg|fedprox, fedprox_mu), experiment (strategy: flat|hier_static|adaptive, model_distribution: global|cluster, seed, run_name).
3. src/config.py: load config.yaml into a frozen dataclass (or pydantic) with validation.
4. A set_seed(seed) utility seeding python, numpy, torch (incl. cudnn deterministic) and exposing a Flower-compatible seed.
5. DESIGN.md summarizing the one-paragraph spec, the three systems (A/B/C), the logical-hierarchy-inside-Strategy decision, the similarity/peer-selection approach, and the config knobs. This file is the source of truth other phases reference.
6. .gitignore (results/, data/, __pycache__, venv).

Folders to create: src/{data,models,strategies,peers,metrics,viz}, tests, results.
```

**Files:** `requirements.txt`, `config.yaml`, `src/config.py`, `DESIGN.md`, `.gitignore`, empty package dirs.
**Check:** `pip install -r requirements.txt` succeeds; `python -c "from src.config import load_config; print(load_config('config.yaml'))"` prints the parsed config.

---

## Phase 1 — Non-IID data partitioning with latent groups

**Goal:** partition the dataset across clients with (a) a tunable Dirichlet non-IID knob and (b) an injected latent-group structure that gives ground-truth cluster labels for evaluation.

**Prompt:**
```
Read DESIGN.md. Implement src/data/partition.py.

Context: the anchor HFL paper (Abad et al.) evaluated only IID data and named non-IID as future work. This partitioner IS the contribution — do not let it degrade to an IID or trivially-balanced split.

Provide a function partition_dataset(cfg) -> (client_datasets, client_group_labels, test_loader) where:
- It loads the dataset named in cfg (FMNIST default; support MNIST and CIFAR-10) via torchvision, normalized.
- It first assigns each of num_clients to one of num_latent_groups. Each group has a biased label support (e.g. 3 groups whose dominant label sets are roughly disjoint) so that clients in the same group have genuinely similar distributions. Return these group ids as the ground-truth cluster labels (for ARI/NMI later).
- Within/across groups it applies Dirichlet(alpha) label skew so the non-IID strength is controllable; lower alpha = more skew.
- It returns a per-client torch Subset/DataLoader and a shared held-out test loader.
- Add a function plot_label_distribution(client_datasets, group_labels, out_path) that saves a stacked-bar heatmap of label counts per client, ordered by group, so the skew is visually obvious.

Add tests/test_partition.py asserting: every sample assigned exactly once, client count correct, lower alpha yields higher average per-client label imbalance (e.g. measured by entropy), and group labels have the expected cardinality.
```

**Files:** `src/data/partition.py`, `tests/test_partition.py`.
**Check:** tests pass; the label-distribution figure shows clear per-group skew that intensifies as `alpha` drops.

---

## Phase 2 — Model + Flower client

**Goal:** a small CNN and a `NumPyClient` with clean parameter (de)serialization.

**Prompt:**
```
Read DESIGN.md. Implement src/models/cnn.py and src/client.py.

cnn.py:
- A small CNN suitable for 28x28 grayscale (FMNIST/MNIST) and a variant for 32x32x3 (CIFAR-10), selected by cfg.model.name.
- get_parameters(model) -> list[np.ndarray] and set_parameters(model, params) helpers (ordered, matching state_dict).
- A helper that returns the parameter index range of the classifier / last layer only, so peers/similarity.py can extract last-layer deltas without re-deriving shapes.

client.py:
- A FlowerClient(NumPyClient) holding a client id, its DataLoader, the model, and cfg.
- fit(): set params, train local_epochs, return updated params + num_examples + a metrics dict including the client id (needed so the strategy can map results back to clients).
- evaluate(): return loss and accuracy on local data.
- A client_fn(context)/client factory compatible with flwr simulation that wires client id -> its partition.

Keep training in a pure train_one_epoch(model, loader, optimizer) so it's unit-testable.
```

**Files:** `src/models/cnn.py`, `src/client.py`.
**Check:** instantiate a client, run one `fit`, confirm params change and loss is finite; param round-trip `set(get(model))` is identity.

---

## Phase 3 — Baseline A: flat FedAvg + logging

**Goal:** a runnable flat FedAvg run that logs per-round global accuracy/loss to CSV. This is the number to beat and proves the simulation plumbing works.

**Prompt:**
```
Read DESIGN.md. Implement src/strategies/flat_fedavg.py, src/metrics/logging.py, and src/run.py.

logging.py: a RoundLogger that accumulates rows {run_name, strategy, round, global_acc, global_loss, num_groups, ari, nmi, wall_time} into a pandas DataFrame and writes results/<run_name>.csv. Fields that don't apply to a strategy stay null.

flat_fedavg.py: subclass flwr FedAvg. After each round, evaluate the aggregated global model on the shared test set (centralized evaluation via evaluate_fn) and push a row to the RoundLogger.

run.py: entrypoint that loads config, partitions data, builds client_fn, selects the strategy by cfg.experiment.strategy, runs flwr simulation (start_simulation / run_simulation — use the API matching the pinned flwr version), and on completion writes the CSV. For now only the 'flat' branch needs to work.

Run it for a short config (e.g. 10 rounds, 20 clients) and paste the resulting CSV head and final accuracy.
```

**Files:** `src/strategies/flat_fedavg.py`, `src/metrics/logging.py`, `src/run.py`.
**Check:** a CSV appears under `results/`, accuracy rises over rounds, run completes without errors.

---

## Phase 4 — Similarity + peer selection core (pure, tested)

**Goal:** the heart of the contribution, built as **pure functions** independent of Flower so you can unit-test the math before wiring it in.

**Prompt:**
```
Read DESIGN.md. Implement src/peers/similarity.py and src/peers/selection.py as pure functions (numpy/sklearn/networkx in, assignments out — no Flower imports).

similarity.py:
- compute_deltas(client_params: dict[cid, list[np.ndarray]], global_params, source: 'last_layer'|'full', last_layer_slice) -> dict[cid, np.ndarray] returning a flattened delta vector per client.
- cosine_similarity_matrix(deltas) -> (ordered_cids, np.ndarray) using sklearn.metrics.pairwise.cosine_similarity.

selection.py:
- build_peer_graph(cids, sim_matrix, threshold, top_k) -> networkx.Graph with similarity edge weights, sparsified by threshold AND/OR top-k neighbors per node.
- select_peers(graph, method: 'louvain'|'greedy_modularity'|'spectral_k', k=None) -> dict[cid, group_id], the adaptive client->edge-aggregator assignment. Spectral/agglomerative with fixed k is the ablation variant.
- apply_hysteresis(prev_assignment, new_assignment, ...) so groups don't reshuffle every round.

tests/test_selection.py: construct synthetic deltas with planted block structure (e.g. 3 clear clusters) and assert the recovered communities match the planted labels with high ARI; assert threshold/top-k change graph density as expected.
```

**Files:** `src/peers/similarity.py`, `src/peers/selection.py`, `tests/test_selection.py`.
**Check:** on planted-cluster synthetic deltas, recovered grouping ARI ≈ 1.0; varying threshold visibly changes edge count.

---

## Phase 5 — Hierarchical aggregation strategies (B and C)

**Goal:** the custom strategies that perform intra-group → cross-group aggregation following Abad et al.'s Algorithm 2 (intra-cluster every round, global sync every `H` rounds). B uses fixed groups; C calls Phase 4 to recompute groups adaptively at sync boundaries.

**Prompt:**
```
Read DESIGN.md and src/peers/*. Implement src/strategies/hier_static.py and src/strategies/adaptive_peers.py. Factor shared hierarchical-aggregation math into a small helper. Follow Abad et al. Algorithm 2: aggregate within clusters every round, average across clusters only on global-sync rounds.

Shared helper hierarchical_aggregate(results_by_cid, assignment, weights, is_sync_round) -> (global_params | None, {group_id: group_params}):
- For each group: FedAvg the member clients' params weighted by num_examples (edge/cluster model).
- If is_sync_round: aggregate the cluster models into a global model (weighted by group size or num_examples) and return it; otherwise return None for global and keep cluster models separate.

A round is a global-sync round when round % hierarchy.global_sync_every (H) == 0.

Distribution rule in configure_fit:
- On the round AFTER a sync, or when model_distribution == 'global', send the global model.
- Otherwise send each client its current cluster model (this is what creates the intra-cluster locality between syncs).

hier_static.py (Baseline B): subclass FedAvg/Strategy. Assignment fixed at init (random or round-robin into hierarchy.num_groups) — the static analog of Abad et al.'s distance-based clustering. Use hierarchical_aggregate with the H schedule. Log num_groups.

adaptive_peers.py (System C): same H schedule, but:
  1. Extract per-client params from results (use the client id returned in fit metrics).
  2. If round <= warmup_rounds: random/flat grouping.
  3. Else if it's a global-sync round (and round % recluster_every == 0): compute deltas vs the last global model, cosine matrix, build_peer_graph, select_peers, apply_hysteresis -> new assignment. Otherwise reuse the last assignment.
  4. hierarchical_aggregate with is_sync_round.
  5. Apply the distribution rule above.
  6. On sync rounds, evaluate the global model centrally; always compute ARI and NMI of the current assignment vs the ground-truth group labels from Phase 1 and log them.

Extend run.py to route 'hier_static' and 'adaptive'. Run a short adaptive config (e.g. H=2) and paste the CSV showing ari/nmi climbing as accuracy improves.
```

**Files:** `src/strategies/hier_static.py`, `src/strategies/adaptive_peers.py`, updated `src/run.py`.
**Check:** adaptive run logs rising `ari`/`nmi`; with strong non-IID, C's accuracy curve is above B and A; global accuracy is only logged on sync rounds.

---

## Phase 6 — Visualization (NetworkX peer graph + matplotlib curves)

**Goal:** the figures that sell the result.

**Prompt:**
```
Read DESIGN.md. Implement src/viz/graph.py and src/viz/curves.py.

graph.py: render_peer_graph(graph, assignment, round, out_path): draw the NetworkX client graph with nodes colored by current community/edge aggregator, edge alpha proportional to similarity weight, layout stable across rounds (fixed spring layout seed) so snapshots are comparable. Add a helper to dump a snapshot every recluster round and optionally stitch them into a GIF showing groups forming/merging over training.

curves.py:
- plot_accuracy(csvs, out_path): overlay global_acc vs round for runs A/B/C from their CSVs.
- plot_cluster_quality(csv, out_path): ari & nmi vs round for the adaptive run.
- plot_convergence_table -> save a results table (rounds-to-target-accuracy, final acc, final ari) to results/summary.csv and a rendered PNG/markdown table via pandas.

Generate all figures from the Phase 5 short runs and list the saved paths.
```

**Files:** `src/viz/graph.py`, `src/viz/curves.py`.
**Check:** peer-graph snapshots show colored communities that stabilize over rounds; A/B/C accuracy overlay renders.

---

## Phase 7 — Experiment runner, sweeps, evaluation

**Goal:** reproducible multi-run experiments and the comparison tables your report needs.

**Prompt:**
```
Read DESIGN.md. Add scripts/experiments.py that runs a sweep over: strategy ∈ {flat, hier_static, adaptive}, alpha ∈ {0.1, 0.5, 1.0}, H (global_sync_every) ∈ {1, 2, 4}, model_distribution ∈ {global, cluster}, and 3 seeds each. Each run reuses run.py with a generated config; results go to results/<run_name>.csv.

Then add scripts/aggregate_results.py that loads all run CSVs and produces:
- A master results table (pandas): mean ± std final accuracy and final ARI per (strategy, alpha, H, distribution).
- Convergence metric: rounds to reach X% of flat-baseline's final accuracy.
- A communication-cost proxy in the spirit of Abad et al.'s latency analysis: count client<->edge transfers (cheap, every round) separately from edge<->global transfers (expensive, every H rounds), and report an improvement factor vs flat FedAvg. Show the accuracy-vs-H trade-off (higher H = fewer global syncs = lower cost, watch accuracy).
Save as results/master_table.csv and a markdown version for the report.

Keep runs short enough to finish on CPU/single GPU; expose num_rounds and num_clients as CLI overrides. Run a reduced sweep (e.g. alpha=0.1 only, 2 seeds) end-to-end and paste master_table.csv.
```

**Files:** `scripts/experiments.py`, `scripts/aggregate_results.py`.
**Check:** master table populates with mean±std across seeds; adaptive shows its largest gains at `alpha=0.1`.

---

## Phase 8 — README + report assets

**Goal:** make it reproducible and submission-ready.

**Prompt:**
```
Read DESIGN.md and the results so far. Write README.md: problem statement, a short related-work paragraph positioning the work as a non-IID, data-similarity-based adaptive-assignment extension of Abad et al. (ICASSP 2020) and an instance of the dynamic UE-assignment problem framed by Rana et al., method (adaptive peer-selection-as-community-detection with the H-periodic global sync and the logical-hierarchy-in-Strategy note), repo map, exact commands to reproduce each figure/table, and a short results paragraph referencing the generated figures. Cite both anchor papers and FedAvg. Add a Makefile or run.sh with targets: install, test, baseline, adaptive, sweep, figures. Ensure `make test` runs the unit tests and a tiny smoke simulation.
```

**Files:** `README.md`, `Makefile`/`run.sh`.
**Check:** a fresh clone + `make install && make test && make figures` reproduces the headline outputs.

---

## Evaluation: what to actually report
- **Task:** global test accuracy + per-cluster accuracy; rounds-to-target (convergence speed); final accuracy vs A/B/C across `alpha`.
- **Cluster recovery:** ARI / NMI of discovered communities vs ground-truth latent groups, over rounds — the figure that shows peer selection *learning the structure*.
- **Stability:** assignment churn per round (fraction of clients changing group); shows hysteresis/warm-up working.
- **Cost:** communication proxy; note HFL's locality advantage even if simulated.
- **Ablations:** similarity source (last-layer vs full), `recluster_every`, threshold/top-k, community method, fixed-K vs emergent-K, global vs cluster distribution, and **global-sync cadence `H`** (the accuracy/communication trade-off — your direct comparison point to Abad et al.).

## Gotchas (flag these to Cursor when they bite)
- **Mapping results back to clients:** Flower's `aggregate_fit` gives you `(ClientProxy, FitRes)` pairs — you must return the client id inside `fit`'s metrics dict to key the similarity matrix. Don't rely on proxy ordering.
- **Flower API drift:** 1.x changed `start_simulation` → `run_simulation`/`ServerApp`. Pin the version and let Cursor match one API; don't mix.
- **Early-round noise:** without warm-up, round-1 deltas cluster garbage. Keep `warmup_rounds ≥ 3`.
- **Delta dimensionality:** full-model cosine is dominated by shared backbone and washes out signal — default to last-layer deltas; expose full as a switch for the ablation.
- **Personalized eval:** if `model_distribution=cluster`, "global accuracy" is ambiguous — report both a centralized global-model accuracy and the mean per-cluster accuracy.

## Stretch (only if time allows)
- Replace community detection with **top-k personalized peer graphs** (each client aggregates with its k nearest peers — true decentralized peer selection, closer to your thesis framing) and compare against the edge-aggregator formulation.
- Add **client dropout / stragglers** and show adaptive grouping degrades more gracefully than static.
- Swap planted groups for **rotated-MNIST** latent structure (covariate shift instead of label skew) to show the method isn't label-skew-specific.

---

## References (for the report bibliography)

**Anchor papers (uploaded):**
- M. S. H. Abad, E. Ozfatura, D. Gündüz, O. Ercetin. *Hierarchical Federated Learning Across Heterogeneous Cellular Networks.* ICASSP 2020, pp. 8866–8870. — The 3-tier HFL loop (Algorithm 2), the `H` global-sync cadence, distance-based clustering, IID CIFAR-10/ResNet18 results. We extend it to non-IID with adaptive, similarity-based assignment.
- O. Rana, T. Spyridopoulos, N. Hudson, M. Baughman, K. Chard, I. Foster, A. Khan. *Hierarchical and Decentralised Federated Learning.* — Frames aggregation-node and UE assignment as a dynamic placement problem (HierFedML) and the generalisation-vs-personalisation trade-off under non-IID locality.

**Methods referenced by the design:**
- B. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg). AISTATS 2017.
- T. Li et al. *Federated Optimization in Heterogeneous Networks* (FedProx). 2018. — optional local-side correction for the heterogeneity ablation.
- S. Karimireddy et al. *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.* ICML 2020. — alternative drift-correction.
- F. Sattler, S. Wiedemann, K.-R. Müller, W. Samek. *Robust and Communication-Efficient Federated Learning from Non-IID Data* (clustered FL). 2019. — closest precedent for the cosine-similarity-of-updates clustering mechanism (cited within Abad et al.).
- Z. Xu, D. Zhao, W. Liang, O. Rana et al. *HierFedML: Aggregator Placement and UE Assignment for Hierarchical Federated Learning in Mobile Edge Computing.* IEEE TPDS. — the UE-assignment problem we instantiate.
- Personalised FL for the cluster-model variant: M. G. Arivazhagan et al. *FedPer* (2019); Y. Deng, M. Kamani, M. Mahdavi. *APFL* (2020); C. Dinh, N. Tran, T. D. Nguyen. *pFedMe* (NeurIPS 2020).

**One-line novelty for your abstract:** *We replace the static, distance-based client-to-aggregator assignment of hierarchical FL with an adaptive, data-similarity-driven peer selection that recovers latent client structure under non-IID data, improving accuracy over flat and statically-clustered HFL at comparable communication cost.*