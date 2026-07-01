"""Catalog ingestion from SHL's published JSON endpoint."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from app.core.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_directories
from app.models.catalog import CatalogDataset, CatalogItem
from app.utils.text import normalize_whitespace


LOGGER = logging.getLogger(__name__)

DEFAULT_SOURCE_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
DEFAULT_RAW_CACHE = RAW_DATA_DIR / "shl_product_catalog.raw.json"
DEFAULT_PROCESSED_CACHE = PROCESSED_DATA_DIR / "shl_product_catalog.normalized.json"
DEFAULT_CACHE_TTL = timedelta(hours=24)


@dataclass(slots=True)
class ScrapeArtifacts:
    """Paths written by the scraper."""

    raw_path: Path
    processed_path: Path


class CatalogScraperError(RuntimeError):
    """Raised when the catalog cannot be downloaded or parsed."""


class SHLCatalogScraper:
    """Fetch, normalize, cache, and export the SHL catalog."""

    def __init__(
        self,
        source_url: str = DEFAULT_SOURCE_URL,
        raw_cache_path: Path = DEFAULT_RAW_CACHE,
        processed_cache_path: Path = DEFAULT_PROCESSED_CACHE,
        cache_ttl: timedelta = DEFAULT_CACHE_TTL,
        session: requests.Session | None = None,
    ) -> None:
        self.source_url = source_url
        self.raw_cache_path = raw_cache_path
        self.processed_cache_path = processed_cache_path
        self.cache_ttl = cache_ttl
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                )
            }
        )

    def scrape(self, force_refresh: bool = False) -> CatalogDataset:
        """Return a normalized dataset, using cache when it is still fresh."""

        ensure_project_directories()

        cached = self._load_cached_dataset()
        if cached is not None and not force_refresh:
            LOGGER.info("Loaded catalog from cache: %s", self.processed_cache_path)
            return cached

        raw_text = self._fetch_raw_text(force_refresh=force_refresh)
        records = self._parse_and_normalize(raw_text)
        dataset = CatalogDataset.from_items(source_url=self.source_url, items=records)
        self._write_cache(raw_text=raw_text, dataset=dataset)
        return dataset

    def _load_cached_dataset(self) -> CatalogDataset | None:
        if not self.processed_cache_path.exists():
            return None
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            self.processed_cache_path.stat().st_mtime, tz=timezone.utc
        )
        if age > self.cache_ttl:
            return None
        try:
            payload = json.loads(self.processed_cache_path.read_text(encoding="utf-8"))
            return CatalogDataset.model_validate(payload)
        except Exception as exc:  # pragma: no cover - cache corruption is rare
            LOGGER.warning("Failed to load cached catalog: %s", exc)
            return None

    def _fetch_raw_text(self, force_refresh: bool = False) -> str:
        if self.raw_cache_path.exists() and not force_refresh:
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                self.raw_cache_path.stat().st_mtime, tz=timezone.utc
            )
            if age <= self.cache_ttl:
                return self.raw_cache_path.read_text(encoding="utf-8")

        response = self.session.get(self.source_url, timeout=30)
        response.raise_for_status()
        raw_text = response.text
        self.raw_cache_path.write_text(raw_text, encoding="utf-8")
        return raw_text

    def _parse_and_normalize(self, raw_text: str) -> list[CatalogItem]:
        try:
            payload = json.loads(raw_text, strict=False)
        except json.JSONDecodeError as exc:
            raise CatalogScraperError(f"Unable to parse catalog JSON: {exc}") from exc

        if not isinstance(payload, list):
            raise CatalogScraperError("Catalog payload must be a top-level JSON array.")

        items: list[CatalogItem] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            item = CatalogItem.from_source(raw)
            if item.status and normalize_whitespace(item.status).lower() != "ok":
                continue
            items.append(item)

        if not items:
            raise CatalogScraperError("No usable catalog records were found.")
        return items

    def _write_cache(self, *, raw_text: str, dataset: CatalogDataset) -> ScrapeArtifacts:
        self.raw_cache_path.write_text(raw_text, encoding="utf-8")
        self.processed_cache_path.write_text(
            dataset.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return ScrapeArtifacts(raw_path=self.raw_cache_path, processed_path=self.processed_cache_path)


def scrape_catalog(force_refresh: bool = False) -> CatalogDataset:
    """Convenience wrapper for scripts and future service startup hooks."""

    return SHLCatalogScraper().scrape(force_refresh=force_refresh)


def main() -> None:
    """Run the scraper as a standalone utility."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    dataset = scrape_catalog(force_refresh=False)
    LOGGER.info(
        "Catalog ready: %s total items, %s recommendable items",
        dataset.total_items,
        dataset.recommendable_items,
    )


if __name__ == "__main__":
    main()
