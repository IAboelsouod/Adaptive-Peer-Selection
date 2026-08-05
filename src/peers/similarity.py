"""Client update deltas and cosine similarity (pure numpy/sklearn)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SimilaritySource = Literal["last_layer", "full"]


def _flatten_params(params: list[np.ndarray], start: int = 0, end: int | None = None) -> np.ndarray:
    """Concatenate selected parameter arrays into one vector."""
    sliced = params[start:end]
    return np.concatenate([array.ravel() for array in sliced])


def compute_deltas(
    client_params: dict[int, list[np.ndarray]],
    global_params: list[np.ndarray],
    source: SimilaritySource,
    last_layer_slice: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Return flattened update deltas ``w_i - w_global`` per client.

    Parameters
    ----------
    client_params:
        Mapping from client id to model weight arrays (``get_parameters`` order).
    global_params:
        Global model weights used as the reference for the delta.
    source:
        ``full`` uses every layer; ``last_layer`` uses ``last_layer_slice`` only
        (from ``classifier_param_indices`` in ``src.models.cnn``).
    last_layer_slice:
        ``(start, end)`` parameter-array indices for the classifier / last layer.
    """
    if source not in {"last_layer", "full"}:
        raise ValueError(f"Unsupported similarity source: {source!r}")

    start, end = last_layer_slice
    global_flat = (
        _flatten_params(global_params)
        if source == "full"
        else _flatten_params(global_params, start, end)
    )

    deltas: dict[int, np.ndarray] = {}
    for client_id, weights in client_params.items():
        client_flat = (
            _flatten_params(weights)
            if source == "full"
            else _flatten_params(weights, start, end)
        )
        deltas[client_id] = client_flat - global_flat
    return deltas


def cosine_similarity_matrix(
    deltas: dict[int, np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """Pairwise cosine similarity of delta vectors.

    Returns
    -------
    ordered_cids:
        Client ids in ascending order (row/column order of the matrix).
    matrix:
        Square similarity matrix aligned with ``ordered_cids``.
    """
    if not deltas:
        return [], np.empty((0, 0), dtype=np.float64)

    ordered_cids = sorted(deltas.keys())
    stacked = np.vstack([deltas[cid] for cid in ordered_cids])
    matrix = cosine_similarity(stacked)
    return ordered_cids, matrix
