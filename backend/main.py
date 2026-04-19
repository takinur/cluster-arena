"""
FastAPI backend — exposes clustering results as JSON so the React
dashboard can fetch and render them without touching Python directly.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from clustering import ALGORITHMS, ClusterResult, run_algorithm, run_all

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Clustering Benchmark API",
    description="Run and compare 8 clustering algorithms on 2-D data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data — generate synthetic blobs that match the Colab notebook shape
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
DATA: np.ndarray = np.load("data.npy") if __import__("pathlib").Path("data.npy").exists() else (
    # Fallback: reproducible synthetic data with similar statistical structure
    np.vstack([
        rng.normal(loc, 0.25, (150, 2))
        for loc in [(0, 0), (2, 0), (1, 1.5), (0.5, -1.2)]
    ])
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Point(BaseModel):
    x: float
    y: float


class ClusterResponse(BaseModel):
    algorithm: str
    n_clusters: int
    elapsed_ms: float
    params: dict
    points: list[Point]
    labels: list[int]


def _build_response(result: ClusterResult) -> ClusterResponse:
    points = [Point(x=float(r[0]), y=float(r[1])) for r in DATA]
    return ClusterResponse(
        algorithm=result.algorithm,
        n_clusters=result.n_clusters,
        elapsed_ms=result.elapsed_ms,
        params=result.params,
        points=points,
        labels=result.labels,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/algorithms", summary="List all available algorithms")
def list_algorithms() -> list[str]:
    return list(ALGORITHMS.keys())


@app.get("/cluster/{algorithm}", response_model=ClusterResponse, summary="Run one algorithm")
def cluster_single(algorithm: str) -> ClusterResponse:
    try:
        result = run_algorithm(algorithm, DATA)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _build_response(result)


@app.get("/cluster", response_model=list[ClusterResponse], summary="Benchmark all algorithms")
def cluster_all() -> list[ClusterResponse]:
    results = run_all(DATA)
    return [_build_response(r) for r in results]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "data_points": len(DATA)}
