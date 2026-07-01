"""FAISS-backed vector store for catalog retrieval."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np

from app.core.paths import PROCESSED_DATA_DIR, VECTOR_STORE_DIR, ensure_project_directories
from app.models.catalog import CatalogDataset, CatalogItem
from app.retrieval.embeddings import (
    EmbeddingBackend,
    HybridTfidfEmbeddingBackend,
    create_embedding_backend,
)


DEFAULT_INDEX_DIR = VECTOR_STORE_DIR / "catalog_index"
DEFAULT_ITEMS_PATH = DEFAULT_INDEX_DIR / "items.json"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "index.faiss"
DEFAULT_BACKEND_PATH = DEFAULT_INDEX_DIR / "backend.pkl"
DEFAULT_MANIFEST_PATH = DEFAULT_INDEX_DIR / "manifest.json"
DEFAULT_DATASET_PATH = PROCESSED_DATA_DIR / "shl_product_catalog.normalized.json"


@dataclass(slots=True)
class CatalogMatch:
    """Search result for a catalog record."""

    item: CatalogItem
    score: float
    rank: int


@dataclass(slots=True)
class VectorStoreManifest:
    """Metadata for persisted vector stores."""

    source_url: str
    backend_name: str
    dimension: int
    item_count: int
    recommendable_count: int
    created_at: str


class CatalogVectorStore:
    """Persisted catalog index with embeddings and a FAISS similarity search."""

    def __init__(
        self,
        *,
        items: list[CatalogItem],
        backend: EmbeddingBackend,
        index: faiss.Index,
        manifest: VectorStoreManifest,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ) -> None:
        self.items = items
        self.backend = backend
        self.index = index
        self.manifest = manifest
        self.index_dir = index_dir

    @property
    def dimension(self) -> int:
        return self.manifest.dimension

    @classmethod
    def build(
        cls,
        dataset: CatalogDataset,
        *,
        backend: EmbeddingBackend | None = None,
        recommendable_only: bool = True,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ) -> "CatalogVectorStore":
        ensure_project_directories()
        index_dir.mkdir(parents=True, exist_ok=True)

        items = [item for item in dataset.items if item.recommendable] if recommendable_only else list(dataset.items)
        if not items:
            raise ValueError("No catalog items available for vector indexing.")

        backend = backend or create_embedding_backend("auto")
        documents = [item.search_text for item in items]
        backend.fit(documents)
        vectors = backend.encode(documents)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embedding backend returned an invalid vector matrix.")

        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        faiss.normalize_L2(vectors)
        index.add(vectors)
        manifest = VectorStoreManifest(
            source_url=dataset.source_url,
            backend_name=getattr(backend, "name", backend.__class__.__name__),
            dimension=int(vectors.shape[1]),
            item_count=len(items),
            recommendable_count=len(items),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        store = cls(items=items, backend=backend, index=index, manifest=manifest, index_dir=index_dir)
        store.save(index_dir)
        return store

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR) -> "CatalogVectorStore":
        manifest_path = index_dir / DEFAULT_MANIFEST_PATH.name
        index_path = index_dir / DEFAULT_INDEX_PATH.name
        items_path = index_dir / DEFAULT_ITEMS_PATH.name
        backend_path = index_dir / DEFAULT_BACKEND_PATH.name

        manifest = VectorStoreManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
        items_payload = json.loads(items_path.read_text(encoding="utf-8"))
        items = [CatalogItem.model_validate(payload) for payload in items_payload]

        backend_name = manifest.backend_name
        if backend_name == "hybrid_tfidf":
            backend = HybridTfidfEmbeddingBackend.load(backend_path)
        else:
            # Optional sentence-transformer backends persist only their config.
            from app.retrieval.embeddings import SentenceTransformerEmbeddingBackend

            backend = SentenceTransformerEmbeddingBackend.load(backend_path)

        index = faiss.read_index(str(index_path))
        return cls(items=items, backend=backend, index=index, manifest=manifest, index_dir=index_dir)

    @classmethod
    def load_or_build(
        cls,
        dataset: CatalogDataset,
        *,
        backend: EmbeddingBackend | None = None,
        recommendable_only: bool = True,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ) -> "CatalogVectorStore":
        manifest_path = index_dir / DEFAULT_MANIFEST_PATH.name
        items_path = index_dir / DEFAULT_ITEMS_PATH.name
        index_path = index_dir / DEFAULT_INDEX_PATH.name
        backend_path = index_dir / DEFAULT_BACKEND_PATH.name
        if manifest_path.exists() and items_path.exists() and index_path.exists() and backend_path.exists():
            try:
                return cls.load(index_dir=index_dir)
            except Exception:
                pass
        return cls.build(
            dataset,
            backend=backend,
            recommendable_only=recommendable_only,
            index_dir=index_dir,
        )

    def save(self, index_dir: Path | None = None) -> None:
        index_dir = index_dir or self.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / DEFAULT_INDEX_PATH.name))
        (index_dir / DEFAULT_ITEMS_PATH.name).write_text(
            json.dumps([item.model_dump(mode="json") for item in self.items], indent=2),
            encoding="utf-8",
        )

        backend_path = index_dir / DEFAULT_BACKEND_PATH.name
        self.backend.save(backend_path)

        (index_dir / DEFAULT_MANIFEST_PATH.name).write_text(
            json.dumps(
                {
                    "source_url": self.manifest.source_url,
                    "backend_name": self.manifest.backend_name,
                    "dimension": self.manifest.dimension,
                    "item_count": self.manifest.item_count,
                    "recommendable_count": self.manifest.recommendable_count,
                    "created_at": self.manifest.created_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def search(self, query: str, top_k: int = 10, oversample: int = 50) -> list[CatalogMatch]:
        """Return the nearest catalog items for a user query."""

        if top_k <= 0:
            return []

        query_vector = self.backend.encode([query])
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("Query embedding must be a single 2D vector.")
        faiss.normalize_L2(query_vector)

        limit = min(max(top_k, oversample), len(self.items))
        scores, indices = self.index.search(query_vector, limit)

        matches: list[CatalogMatch] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.items):
                continue
            matches.append(CatalogMatch(item=self.items[int(idx)], score=float(score), rank=rank))
            if len(matches) >= top_k:
                break
        return matches


def build_catalog_vector_store(
    dataset: CatalogDataset,
    *,
    backend: EmbeddingBackend | None = None,
    recommendable_only: bool = True,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> CatalogVectorStore:
    """Build and persist the FAISS index for the catalog."""

    return CatalogVectorStore.load_or_build(
        dataset,
        backend=backend,
        recommendable_only=recommendable_only,
        index_dir=index_dir,
    )


def load_catalog_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> CatalogDataset:
    """Load a cached normalized catalog dataset from disk."""

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return CatalogDataset.model_validate(payload)
