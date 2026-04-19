# Clustering Algorithm Benchmark

> **8 unsupervised clustering algorithms, one dataset, one interactive dashboard.**  
> Explored in a Google Colab notebook, then productionised with a FastAPI backend and a React frontend.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-f7931e?logo=scikitlearn)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this project does

This project benchmarks and visualises the following clustering algorithms on identical 2-D data so results are directly comparable:

| Algorithm | Library | Key hyper-parameter |
|---|---|---|
| K-Means | scikit-learn | `n_clusters=4`, `init='k-means++'` |
| Affinity Propagation | scikit-learn | `convergence_iter=12` |
| DBSCAN | scikit-learn | `eps=0.2`, `min_samples=5` |
| BIRCH | scikit-learn | `n_clusters=5` |
| OPTICS | scikit-learn | `min_samples=3`, `xi=0.04` |
| **HDBSCAN** | hdbscan | `min_cluster_size=4` |
| Agglomerative | scikit-learn | `linkage='ward'` |
| Spectral | scikit-learn | `affinity='nearest_neighbors'` |

The **Elbow Method** (via Yellowbrick) is used to find the optimal *k* for algorithms that require it.

---

## Project structure

```
.
├── notebook/
│   └── fine-tuning.ipynb      # Original Google Colab exploration
├── backend/
│   ├── main.py                # FastAPI app — REST endpoints
│   ├── clustering.py          # Algorithm registry & benchmark runner
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Dashboard shell + algorithm selector
│   │   └── components/
│   │       └── ClusterChart.jsx  # Per-algorithm scatter chart
│   ├── vite.config.js
│   └── package.json
└── .gitignore
```

---

## Quick start

### 1 — Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
# API docs → http://localhost:8000/docs
```

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
# Dashboard → http://localhost:5173
```

The Vite dev server proxies `/cluster` and `/algorithms` to `:8000` automatically.

### 3 — Notebook

Open `notebook/fine-tuning.ipynb` in [Google Colab](https://colab.research.google.com) or locally with Jupyter. The notebook is self-contained — it installs its own dependencies.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/algorithms` | List all registered algorithm names |
| `GET` | `/cluster` | Run **all** algorithms, return JSON array |
| `GET` | `/cluster/{algorithm}` | Run a single algorithm by name |
| `GET` | `/health` | Liveness check + dataset size |

**Example response** for `/cluster/kmeans`:

```json
{
  "algorithm": "kmeans",
  "n_clusters": 4,
  "elapsed_ms": 12.4,
  "params": { "n_clusters": 4, "init": "k-means++" },
  "points": [{ "x": 0.842, "y": 0.039 }, "..."],
  "labels": [0, 1, 2, 0, 3, "..."]
}
```

---

## Notebook highlights

### Environment & data

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 20, 'linewidths': 3}

data = np.load('/content/data.npy')
print(data[:5])
# [[ 0.84219027  0.03995678]
#  [ 0.57487825 -0.43018898]
#  [-0.33429872  0.85586816]
#  [ 0.75188718  0.56642066]
#  [ 1.9682912   0.45406949]]
```

### Unified benchmarking function

Every algorithm passes through the same wrapper — timing, coloring, and plotting in one call. Noise points (label = -1, from density-based methods) are coloured black.

```python
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors  = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title(f'Clusters found by {algorithm.__name__}', fontsize=24)
    plt.text(-0.5, 0.7,
             f'Clustering took {end_time - start_time:.2f} s',
             fontsize=14)
```

### Elbow Method — finding optimal *k*

```python
from yellowbrick.cluster import KElbowVisualizer

visualizer = KElbowVisualizer(KMeans(init='k-means++'), k=(2, 10))
visualizer.fit(data)
visualizer.show()
```

The same visualizer is reused for **BIRCH** and **Agglomerative** — it's algorithm-agnostic as long as the estimator exposes `fit()`.

### Running all 8 algorithms

```python
from sklearn.cluster import (
    KMeans, DBSCAN, Birch, OPTICS,
    AffinityPropagation, AgglomerativeClustering, SpectralClustering,
)
import hdbscan

# K-Means  ──────────────────────────────────────────────────────────────────
plot_clusters(data, KMeans, (), {'n_clusters': 4, 'init': 'k-means++'})

# Affinity Propagation  ─────────────────────────────────────────────────────
plot_clusters(data, AffinityPropagation, (), {'convergence_iter': 12})

# DBSCAN  ───────────────────────────────────────────────────────────────────
plot_clusters(data, DBSCAN, (), {'eps': 0.2, 'min_samples': 5})

# BIRCH  ────────────────────────────────────────────────────────────────────
plot_clusters(data, Birch, (), {'n_clusters': 5})

# OPTICS  ───────────────────────────────────────────────────────────────────
plot_clusters(data, OPTICS, (), {'min_samples': 3, 'xi': 0.04, 'min_cluster_size': 0.2})

# HDBSCAN  ──────────────────────────────────────────────────────────────────
plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size': 4})

# Agglomerative  ────────────────────────────────────────────────────────────
plot_clusters(data, AgglomerativeClustering, (),
              {'n_clusters': 4, 'affinity': 'euclidean', 'linkage': 'ward'})

# Spectral  ─────────────────────────────────────────────────────────────────
plot_clusters(data, SpectralClustering, (),
              {'n_clusters': 4, 'affinity': 'nearest_neighbors'})
```

---

## Key takeaways

- **HDBSCAN** is the most robust — it discovers clusters of varying density without requiring *k* to be specified upfront, and correctly labels outliers as noise.
- **DBSCAN** is faster but sensitive to `eps`; small changes drastically alter cluster shape.
- **K-Means** converges fastest (`k-means++` init avoids poor centroids) but forces spherical clusters.
- **Spectral** handles non-convex shapes but scales poorly to large *n*.
- **Affinity Propagation** is parameter-light but has O(n²) memory — impractical beyond ~10k points.

---

## Stack

| Layer | Technology |
|---|---|
| Notebook | Python 3.10, scikit-learn, HDBSCAN, Yellowbrick, Matplotlib, Seaborn |
| Backend | FastAPI, Uvicorn, NumPy, Pydantic v2 |
| Frontend | React 18, Recharts, Axios, Vite |

---

## License

MIT — see [LICENSE](LICENSE).
