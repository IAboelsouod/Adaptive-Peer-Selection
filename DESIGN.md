# DESIGN — Adaptive Peer Selection for Hierarchical Federated Learning

This document is the source of truth for terminology and architecture. Read it before implementing any phase.

## One-paragraph spec

A Flower-based FL **simulation** with three logical tiers — clients → edge aggregators → global server — where the **assignment of clients to edge aggregators is recomputed adaptively** (at each global sync) from the **cosine similarity of client model-update deltas**, replacing the static, distance-based assignment of Abad et al. (ICASSP 2020) with a data-driven one under non-IID data. Similar clients become *peers* sharing an edge aggregator; each edge group runs intra-group FedAvg; edge models are then aggregated into a global model. We compare three systems: **(A) flat FedAvg baseline**, **(B) hierarchical FL with static/random groups**, **(C) adaptive peer selection (ours)** — under controllable non-IID data, and we score both **task accuracy** and **cluster recovery quality** against a known latent group structure.

## Three systems

| ID | Strategy key | Description |
|----|--------------|-------------|
| **A** | `flat` | Standard FedAvg — all clients share one global model every round. |
| **B** | `hier_static` | Hierarchical FL with **fixed** client→edge-aggregator assignment (random or round-robin into `hierarchy.num_groups`). Follows Abad et al. Algorithm 2 with `H`-periodic global sync. |
| **C** | `adaptive` | Same HFL loop as B, but assignment is **recomputed at global-sync boundaries** from cosine similarity of update **deltas** → peer graph → community detection. |

## Logical hierarchy inside Strategy

The three-tier hierarchy is **logical**, not physical:

- **No separate edge-server processes.** Everything runs inside a custom Flower `Strategy`.
- `aggregate_fit` receives all client results, groups them by edge aggregator (community), aggregates within each group (intra-cluster FedAvg), then optionally aggregates across groups into a global model (on sync rounds).
- Clients are Flower `NumPyClient` instances in simulation; edge aggregators are data structures inside the strategy.

## Similarity and peer selection

1. **Delta:** per-client update `Δ_i = w_i^(t) − w_global^(t−1)`, flattened. Default: **last-layer / classifier delta** (`peers.similarity_source: last_layer`); full-model delta available for ablation.
2. **Similarity matrix:** pairwise cosine similarity of deltas (`sklearn.metrics.pairwise.cosine_similarity`).
3. **Peer graph:** NetworkX graph; nodes = clients; edge weight = similarity; sparsified by `peers.threshold` and/or `peers.top_k`.
4. **Community detection:** Louvain (default), greedy modularity, or fixed-K spectral/agglomerative (`peers.community_method`). Each community = one **edge aggregator**. Number of groups is **emergent** (except spectral_k ablation).
5. **Warm-up:** first `peers.warmup_rounds` use flat/random grouping (early deltas are noise).
6. **Re-clustering:** at global-sync boundaries when `round % peers.recluster_every == 0`, with optional `peers.hysteresis` to reduce assignment churn.
7. **Client ID in metrics:** each client's `fit()` must return its id in the metrics dict so the strategy can key deltas — do not rely on proxy ordering.

## H-periodic global sync (Abad et al. Algorithm 2)

- **Every round:** clients train locally; edge aggregators FedAvg within their cluster.
- **Every `H` rounds** (`hierarchy.global_sync_every`): global server averages cluster models, then redistributes.
- **Between syncs:** clients train on and receive their **cluster** model (when `experiment.model_distribution: cluster`).
- **On sync round + round after sync:** clients may receive the **global** model (configurable via `model_distribution`).
- **Adaptive re-clustering** happens at sync boundaries when a fresh global reference exists for delta computation.

Higher `H` = fewer global syncs = lower backhaul cost but more inter-cluster drift.

## Model distribution

Controlled by `experiment.model_distribution`:

- **`global`:** clients always receive the global model (pure HFL).
- **`cluster`:** between syncs, clients receive their cluster/edge model (personalized HFL). Usually better under strong non-IID.

## Data partitioning

- Dirichlet `data.alpha` controls label skew (lower = more non-IID).
- `data.num_latent_groups` injects ground-truth cluster structure with disjoint-ish label support per group.
- Ground-truth group labels enable **ARI/NMI** evaluation of peer selection quality.

## Config knobs (all in `config.yaml`)

| Section | Key knobs |
|---------|-----------|
| `dataset` | `name` (fmnist, mnist, cifar10) |
| `federation` | `num_clients`, `num_rounds`, `local_epochs`, `fraction_fit` |
| `hierarchy` | `global_sync_every` (H), `num_groups` (static baseline) |
| `data` | `alpha`, `num_latent_groups`, `partition_scheme` |
| `model` | `name` |
| `peers` | `similarity_source`, `threshold`, `top_k`, `community_method`, `recluster_every`, `warmup_rounds`, `hysteresis` |
| `training` | `lr`, `batch_size`, `optimizer`, `local_optimizer`, `fedprox_mu` |
| `experiment` | `strategy`, `model_distribution`, `seed`, `run_name` |

Nothing is hardcoded in logic — load via `src.config.load_config` and `set_seed`.

## Aggregation math

Keep FedAvg and hierarchical aggregation as **pure numpy functions** (take/return arrays) so they are unit-testable without Flower.

## Logging and results

- `RoundLogger` writes per-round metrics to `results/<run_name>.csv`.
- Fields: `run_name`, `strategy`, `round`, `global_acc`, `global_loss`, `num_groups`, `ari`, `nmi`, `wall_time`.
- Peer-graph snapshots and accuracy curves go under `results/` (gitignored).

## Terminology

| Term | Meaning |
|------|---------|
| **Peer** | A client with similar data/update profile; members of the same community. |
| **Edge aggregator** | Logical cluster head that FedAvgs its member clients each round. |
| **Community** | Output of graph community detection; maps 1:1 to an edge aggregator. |
| **Delta** | Client update relative to the last global model: `w_i − w_global`. |
| **Global sync** | Round where cluster models are merged into the global model (every `H` rounds). |

## References

- Abad et al., ICASSP 2020 — HFL template, Algorithm 2, static distance-based assignment.
- Rana et al. — dynamic UE/aggregator assignment framing.
- McMahan et al., FedAvg — baseline aggregation.
- Sattler et al., clustered FL — cosine-similarity-of-updates precedent.
