"""
Core clustering logic — mirrors the Google Colab notebook.
All algorithms operate on the same 2-D dataset so results are directly comparable.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    Birch,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    SpectralClustering,
)
import hdbscan


@dataclass
class ClusterResult:
    algorithm: str
    labels: list[int]
    n_clusters: int
    elapsed_ms: float
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry of algorithms + their default parameters
# ---------------------------------------------------------------------------
ALGORITHMS: dict[str, tuple[Any, dict]] = {
    "kmeans": (KMeans, {"n_clusters": 4, "init": "k-means++"}),
    "affinity_propagation": (AffinityPropagation, {"convergence_iter": 12}),
    "dbscan": (DBSCAN, {"eps": 0.2, "min_samples": 5}),
    "birch": (Birch, {"n_clusters": 5}),
    "optics": (OPTICS, {"min_samples": 3, "xi": 0.04, "min_cluster_size": 0.2}),
    "hdbscan": (hdbscan.HDBSCAN, {"min_cluster_size": 4}),
    "agglomerative": (
        AgglomerativeClustering,
        {"n_clusters": 4, "metric": "euclidean", "linkage": "ward"},
    ),
    "spectral": (SpectralClustering, {"n_clusters": 4, "affinity": "nearest_neighbors"}),
}


def run_algorithm(name: str, data: np.ndarray, params: dict | None = None) -> ClusterResult:
    """Run a single clustering algorithm and return a structured result."""
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from: {list(ALGORITHMS)}")

    cls, default_params = ALGORITHMS[name]
    effective_params = {**default_params, **(params or {})}

    t0 = time.perf_counter()
    labels = cls(**effective_params).fit_predict(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    n_clusters = len(set(labels) - {-1})  # -1 = noise in DBSCAN / HDBSCAN

    return ClusterResult(
        algorithm=name,
        labels=labels.tolist(),
        n_clusters=n_clusters,
        elapsed_ms=round(elapsed_ms, 2),
        params=effective_params,
    )


def run_all(data: np.ndarray) -> list[ClusterResult]:
    """Benchmark every registered algorithm on the same dataset."""
    return [run_algorithm(name, data) for name in ALGORITHMS]
