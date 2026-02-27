import os
import numpy as np
from typing import List, Protocol, Optional


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...


class SentenceTransformersEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_root: Optional[str] = None,
        offline: bool = False,
    ):
        """
        cache_root: root folder under which we keep:
          - HF hub cache in <cache_root>/hf
          - sentence-transformers cache_folder in <cache_root>/st
        """
        if cache_root is None:
            cache_root = os.path.expanduser("~/.cache/agentanalytics")

        hf_home = os.path.join(cache_root, "hf")
        st_home = os.path.join(cache_root, "st")
        os.makedirs(hf_home, exist_ok=True)
        os.makedirs(st_home, exist_ok=True)

        # Ensure HF uses a persistent cache
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", st_home)

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer

        # IMPORTANT: use the fully-qualified repo id.
        # cache_folder is where SentenceTransformers stores the model folder (with modules.json).
        self._m = SentenceTransformer(model_name, cache_folder=st_home)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, show_progress_bar=False, normalize_embeddings=True))


class TfidfSvdEmbedder:
    def __init__(self, n_components: int = 128, max_features: int = 20000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self._svd = TruncatedSVD(n_components=n_components, random_state=13)

    def embed(self, texts: List[str]) -> np.ndarray:
        X = self._vec.fit_transform(texts)
        Z = self._svd.fit_transform(X)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z


def make_embedder(
    prefer_sentence_transformers: bool = True,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_root: Optional[str] = None,
    offline: bool = False,
):
    if prefer_sentence_transformers:
        try:
            return SentenceTransformersEmbedder(model_name=model_name, cache_root=cache_root, offline=offline)
        except Exception:
            pass
    return TfidfSvdEmbedder()
