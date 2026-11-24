"""Download the official Medicare publications from medicare.gov.

These are works of the U.S. federal government and are in the public domain.
medicare.gov publishes no API for the publication list, so the product-number
to filename mapping in ``config`` was established by fetching each candidate
URL and keeping the ones that returned a PDF.

A manifest with SHA-256 digests is written next to the downloads so a later run
can prove it is working on the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from medicare_rag.config import (
    MEDICARE_PUBLICATIONS,
    PUBLICATION_BASE_URL,
    RAW_DIR,
    USER_AGENT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def ensure_raw_data(raw_dir: Path = RAW_DIR, *, force: bool = False) -> dict[str, Path]:
    """Download every publication; return a map of product number to local path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    resolved: dict[str, Path] = {}

    for product, meta in MEDICARE_PUBLICATIONS.items():
        url = PUBLICATION_BASE_URL + meta["slug"]
        target = raw_dir / f"{product}.pdf"
        if force or not target.exists():
            print(f"[download] {url}")
            _download(url, target)
        else:
            print(f"[download] cached {target.name}")
        manifest[product] = {
            "url": url,
            "title": meta["title"],
            "sha256": _sha256(target),
            "bytes": str(target.stat().st_size),
        }
        resolved[product] = target

    (raw_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return resolved


if __name__ == "__main__":  # pragma: no cover - manual entry point
    for product, path in ensure_raw_data().items():
        print(f"{product}: {path} ({path.stat().st_size:,} bytes)")
