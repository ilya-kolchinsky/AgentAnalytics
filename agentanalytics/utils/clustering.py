from dataclasses import dataclass
from typing import List
import math
import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray              # shape [n], -1 = noise
    unique_labels: List[int]        # sorted labels excluding -1
    noise_share: float


def cluster_embeddings(
    emb: np.ndarray,
    *,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    seed: int = 13,
) -> ClusterResult:
    """
    Prefer HDBSCAN if installed; fallback to KMeans.
    """
    labels = None
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(emb)
    except Exception:
        from sklearn.cluster import KMeans
        n = emb.shape[0]
        k = max(2, int(math.sqrt(n / 2)))
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(emb)

    labels = np.asarray(labels, dtype=int)
    uniq = sorted({int(x) for x in labels.tolist() if int(x) != -1})
    noise_share = float(np.mean(labels == -1))
    return ClusterResult(labels=labels, unique_labels=uniq, noise_share=noise_share)


def pick_representative_index(emb: np.ndarray, idxs: List[int]) -> int:
    """
    emb assumed roughly normalized; choose nearest to centroid by cosine similarity.
    """
    sub = emb[idxs]
    c = sub.mean(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c) + 1e-9)
    dots = (sub @ c.T).reshape(-1)
    best_local = int(np.argmax(dots))
    return idxs[best_local]
