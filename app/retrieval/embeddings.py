"""Embedding backends for catalog documents and user queries."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.utils.text import normalize_whitespace


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot be constructed."""


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface shared by all embedding implementations."""

    name: str
    dimension: int

    def fit(self, texts: Sequence[str]) -> "EmbeddingBackend":
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def save(self, path: Path) -> None:
        ...


def _clean_texts(texts: Sequence[str]) -> list[str]:
    return [normalize_whitespace(text) for text in texts]


@dataclass
class HybridTfidfEmbeddingBackend:
    """Fast local fallback that combines word and character TF-IDF features."""

    name: str = "hybrid_tfidf"
    word_vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=40000,
            stop_words="english",
            sublinear_tf=True,
        )
    )
    char_vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        )
    )
    reducer: TruncatedSVD | None = None
    dimension: int = 0

    def fit(self, texts: Sequence[str]) -> "HybridTfidfEmbeddingBackend":
        cleaned = _clean_texts(texts)
        word_matrix = self.word_vectorizer.fit_transform(cleaned)
        char_matrix = self.char_vectorizer.fit_transform(cleaned)
        combined = hstack([word_matrix, char_matrix]).tocsr()

        max_components = min(256, combined.shape[0] - 1, combined.shape[1] - 1)
        if max_components >= 2:
            self.reducer = TruncatedSVD(n_components=max_components, random_state=42)
            dense = self.reducer.fit_transform(combined)
        else:
            dense = combined.toarray()

        self.dimension = int(dense.shape[1])
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = _clean_texts(texts)
        word_matrix = self.word_vectorizer.transform(cleaned)
        char_matrix = self.char_vectorizer.transform(cleaned)
        combined = hstack([word_matrix, char_matrix]).tocsr()

        if self.reducer is not None:
            dense = self.reducer.transform(combined)
        else:
            dense = combined.toarray()

        dense = np.asarray(dense, dtype=np.float32)
        normalize(dense, norm="l2", axis=1, copy=False)
        return dense

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "HybridTfidfEmbeddingBackend":
        with path.open("rb") as handle:
            backend = pickle.load(handle)
        if not isinstance(backend, HybridTfidfEmbeddingBackend):
            raise EmbeddingBackendError(f"Unexpected backend payload in {path}")
        return backend


@dataclass
class SentenceTransformerEmbeddingBackend:
    """Optional BGE-style backend using sentence-transformers when available."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    name: str = "sentence_transformer"
    dimension: int = 0
    _model: object | None = field(default=None, init=False, repr=False)

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on optional package state
            raise EmbeddingBackendError(
                "sentence-transformers is unavailable in this environment"
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbeddingBackend":
        model = self._load_model()
        sample = self.encode(texts[:1]) if texts else self.encode(["sample"])
        self.dimension = int(sample.shape[1])
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        cleaned = [f"passage: {normalize_whitespace(text)}" for text in texts]
        embeddings = model.encode(cleaned, normalize_embeddings=True, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if not self.dimension:
            self.dimension = int(embeddings.shape[1])
        return embeddings

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "model_name": self.model_name,
                    "dimension": self.dimension,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerEmbeddingBackend":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model_name=payload.get("model_name", "BAAI/bge-small-en-v1.5"),
            dimension=int(payload.get("dimension", 0) or 0),
        )


def create_embedding_backend(preferred: str = "auto") -> EmbeddingBackend:
    """Create the best available embedding backend for this environment."""

    preferred = normalize_whitespace(preferred).lower() or "auto"
    if preferred in {"bge", "sentence_transformer", "sentence-transformer", "auto"}:
        try:
            backend = SentenceTransformerEmbeddingBackend()
            backend._load_model()
            return backend
        except Exception:
            if preferred != "auto":
                raise

    return HybridTfidfEmbeddingBackend()
