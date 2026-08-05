"""NetworkX peer-graph rendering and snapshot/GIF helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flwr.common import ndarrays_to_parameters
from flwr.server.server_config import ServerConfig
from flwr.simulation import start_simulation
from PIL import Image

from src.client import make_client_fn
from src.config import Config, load_config, set_seed
from src.data.partition import partition_dataset
from src.metrics.logging import RoundLogger
from src.models.cnn import create_model, get_parameters
from src.strategies.adaptive_peers import build_adaptive_peers_strategy

LAYOUT_SEED = 42

# Stable node positions keyed by client id (computed once per experiment).
_layout_positions: dict[int, tuple[float, float]] = {}

PeerGraphSnapshotCallback = Callable[[nx.Graph, dict[int, int], int], None]


def reset_layout_cache() -> None:
    """Clear cached spring-layout positions (for tests or fresh runs)."""
    _layout_positions.clear()


def stable_layout(graph: nx.Graph) -> dict[int, tuple[float, float]]:
    """Return fixed node positions so snapshots are comparable across rounds.

    Positions are computed once from a node-only skeleton graph with a fixed
    random seed, then reused even as edge sets change between recluster rounds.
    """
    global _layout_positions
    nodes = sorted(graph.nodes())
    if not _layout_positions:
        skeleton = nx.Graph()
        skeleton.add_nodes_from(nodes)
        pos = nx.spring_layout(skeleton, seed=LAYOUT_SEED)
        _layout_positions = {int(node): (float(x), float(y)) for node, (x, y) in pos.items()}
    missing = [node for node in nodes if node not in _layout_positions]
    if missing:
        raise ValueError(f"Layout cache missing nodes: {missing}")
    return {node: _layout_positions[node] for node in nodes}


def _group_color_map(assignment: dict[int, int], nodes: list[int]) -> list:
    """Map hysteresis-stable group ids to consistent colors."""
    palette = plt.cm.tab10.colors
    group_ids = sorted({assignment[node] for node in nodes if node in assignment})
    gid_to_color = {gid: palette[gid % len(palette)] for gid in group_ids}
    return [gid_to_color[assignment[node]] for node in nodes]


def render_peer_graph(
    graph: nx.Graph,
    assignment: dict[int, int],
    round_num: int,
    out_path: str | Path,
) -> Path:
    """Draw the peer graph with community-colored nodes and weighted edges.

    Edge alpha and width scale with the similarity ``weight`` attribute.
    Layout is fixed across calls via :func:`stable_layout`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = sorted(graph.nodes())
    pos = stable_layout(graph)
    node_colors = _group_color_map(assignment, nodes)

    edges = list(graph.edges(data=True))
    if edges:
        weights = np.array([float(data.get("weight", 0.0)) for _, _, data in edges])
        max_weight = float(weights.max()) if weights.size else 1.0
        if max_weight <= 0:
            max_weight = 1.0
        norm = weights / max_weight
        widths = 0.5 + 2.5 * norm
        alphas = 0.15 + 0.75 * norm
    else:
        widths = []
        alphas = []

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=node_colors,
        node_size=280,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )

    for (left, right, data), width, alpha in zip(edges, widths, alphas, strict=True):
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(left, right)],
            width=width,
            alpha=alpha,
            edge_color="steelblue",
            ax=ax,
        )

    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    num_groups = len({assignment[n] for n in nodes if n in assignment})
    ax.set_title(
        f"Peer graph — round {round_num} ({num_groups} communities, "
        f"{graph.number_of_edges()} edges)"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


@dataclass
class PeerGraphSnapshotDumper:
    """Callback that saves a PNG snapshot on each recluster round."""

    out_dir: Path
    frame_paths: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, graph: nx.Graph, assignment: dict[int, int], round_num: int) -> None:
        path = self.out_dir / f"peer_graph_round_{round_num:03d}.png"
        render_peer_graph(graph, assignment, round_num, path)
        self.frame_paths.append(path)


def dump_recluster_snapshots(
    snapshot_cb: PeerGraphSnapshotCallback,
    cfg: Config,
    *,
    out_dir: str | Path = "results/peer_graphs",
) -> list[Path]:
    """Run an adaptive simulation and invoke ``snapshot_cb`` at each recluster round."""
    reset_layout_cache()
    out_dir = Path(out_dir)
    set_seed(cfg.experiment.seed)

    client_datasets, group_labels, test_loader = partition_dataset(cfg)
    client_fn = make_client_fn(cfg, client_datasets)
    round_logger = RoundLogger(cfg.experiment.run_name, cfg.experiment.strategy)

    model = create_model(cfg)
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy = build_adaptive_peers_strategy(
        cfg,
        test_loader,
        round_logger,
        group_labels,
        initial_parameters,
        peer_graph_snapshot_cb=snapshot_cb,
    )

    server_config = ServerConfig(num_rounds=cfg.federation.num_rounds)
    start_simulation(
        client_fn=client_fn,
        num_clients=cfg.federation.num_clients,
        config=server_config,
        strategy=strategy,
    )
    round_logger.write_csv()
    return sorted(out_dir.glob("peer_graph_round_*.png"))


def stitch_snapshots_to_gif(
    frame_paths: list[str | Path],
    out_path: str | Path,
    *,
    duration_ms: int = 800,
) -> Path:
    """Stitch peer-graph PNG snapshots into an animated GIF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in frame_paths]
    if not paths:
        raise ValueError("No snapshot frames provided for GIF stitching.")

    frames = [Image.open(path).convert("RGBA") for path in paths]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    for frame in frames:
        frame.close()
    return out_path


def generate_peer_graph_figures(
    cfg: Config,
    *,
    snapshot_dir: str | Path = "results/peer_graphs",
    gif_path: str | Path = "results/peer_graph_evolution.gif",
) -> tuple[list[Path], Path, Path]:
    """Run adaptive FL, dump recluster snapshots, and stitch a GIF."""
    reset_layout_cache()
    dumper = PeerGraphSnapshotDumper(Path(snapshot_dir))
    csv_path = _run_adaptive_with_snapshots(cfg, dumper)
    gif = stitch_snapshots_to_gif(dumper.frame_paths, gif_path)
    return dumper.frame_paths, gif, csv_path


def _run_experiment(cfg: Config) -> Path:
    """Run one FL experiment and return the CSV log path."""
    set_seed(cfg.experiment.seed)
    client_datasets, group_labels, test_loader = partition_dataset(cfg)
    client_fn = make_client_fn(cfg, client_datasets)
    round_logger = RoundLogger(cfg.experiment.run_name, cfg.experiment.strategy)

    model = create_model(cfg)
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy_name = cfg.experiment.strategy
    if strategy_name == "flat":
        from src.strategies.flat_fedavg import build_flat_fedavg_strategy

        strategy = build_flat_fedavg_strategy(
            cfg, test_loader, round_logger, initial_parameters
        )
    elif strategy_name == "hier_static":
        from src.strategies.hier_static import build_hier_static_strategy

        strategy = build_hier_static_strategy(
            cfg, test_loader, round_logger, group_labels, initial_parameters
        )
    elif strategy_name == "adaptive":
        strategy = build_adaptive_peers_strategy(
            cfg, test_loader, round_logger, group_labels, initial_parameters
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}")

    server_config = ServerConfig(num_rounds=cfg.federation.num_rounds)
    start_simulation(
        client_fn=client_fn,
        num_clients=cfg.federation.num_clients,
        config=server_config,
        strategy=strategy,
    )
    return round_logger.write_csv()


def run_matched_short_experiments(
    config_path: str | Path = "config.yaml",
    *,
    alpha: float = 0.1,
    seed: int = 42,
    num_rounds: int = 10,
    num_clients: int = 50,
    run_prefix: str = "phase6",
    snapshot_dir: str | Path = "results/peer_graphs",
    gif_path: str | Path = "results/peer_graph_evolution.gif",
) -> tuple[dict[str, Path], list[Path], Path]:
    """Run A/B/C with matched alpha and seed; capture peer-graph snapshots for C."""
    base = load_config(config_path)
    base = replace(
        base,
        federation=replace(base.federation, num_rounds=num_rounds, num_clients=num_clients),
        data=replace(base.data, alpha=alpha),
        experiment=replace(base.experiment, seed=seed),
        hierarchy=replace(base.hierarchy, global_sync_every=2),
    )

    csvs: dict[str, Path] = {}
    for strategy, key in [("flat", "A"), ("hier_static", "B")]:
        cfg = replace(
            base,
            experiment=replace(
                base.experiment,
                strategy=strategy,
                run_name=f"{run_prefix}_{strategy}_a{alpha}_s{seed}",
                seed=seed,
            ),
        )
        csvs[key] = _run_experiment(cfg)

    reset_layout_cache()
    dumper = PeerGraphSnapshotDumper(Path(snapshot_dir))
    cfg_c = replace(
        base,
        experiment=replace(
            base.experiment,
            strategy="adaptive",
            run_name=f"{run_prefix}_adaptive_a{alpha}_s{seed}",
            seed=seed,
        ),
    )
    csvs["C"] = _run_adaptive_with_snapshots(cfg_c, dumper)
    gif = stitch_snapshots_to_gif(dumper.frame_paths, gif_path)
    return csvs, dumper.frame_paths, gif


def _run_adaptive_with_snapshots(
    cfg: Config,
    dumper: PeerGraphSnapshotDumper,
) -> Path:
    """Run adaptive FL once, saving peer-graph snapshots at recluster rounds."""
    set_seed(cfg.experiment.seed)
    client_datasets, group_labels, test_loader = partition_dataset(cfg)
    client_fn = make_client_fn(cfg, client_datasets)
    round_logger = RoundLogger(cfg.experiment.run_name, cfg.experiment.strategy)

    model = create_model(cfg)
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy = build_adaptive_peers_strategy(
        cfg,
        test_loader,
        round_logger,
        group_labels,
        initial_parameters,
        peer_graph_snapshot_cb=dumper,
    )

    server_config = ServerConfig(num_rounds=cfg.federation.num_rounds)
    start_simulation(
        client_fn=client_fn,
        num_clients=cfg.federation.num_clients,
        config=server_config,
        strategy=strategy,
    )
    return round_logger.write_csv()


if __name__ == "__main__":
    from src.viz.curves import generate_all_curves

    csvs, snapshots, gif = run_matched_short_experiments()
    curve_paths = generate_all_curves(csvs)

    print("Phase 6 figures generated:")
    for name, path in sorted({**csvs, **curve_paths, "gif": gif}.items()):
        print(f"  {name}: {path}")
    print("Peer-graph snapshots:")
    for path in snapshots:
        print(f"  {path}")
