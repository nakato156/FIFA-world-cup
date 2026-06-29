#!/usr/bin/env python3
"""Descarga y registra los cuatro datasets obligatorios de Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASETS = {
    "international_results": "martj42/international-football-results-from-1872-to-2017",
    "fifa_rankings": "cashncarry/fifaworldranking",
    "fifa_players": "stefanoleone992/ea-sports-fc-24-complete-player-dataset",
    "world_cups": "piterfm/fifa-football-world-cup",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_token_from_dotenv() -> None:
    """Carga solo KAGGLE_API_TOKEN sin ejecutar el contenido de .env."""
    dotenv = ROOT / ".env"
    if "KAGGLE_API_TOKEN" in os.environ or not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("KAGGLE_API_TOKEN="):
            token = line.split("=", 1)[1].strip().strip("\"'")
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
            return


def main() -> None:
    load_token_from_dotenv()
    if not os.environ.get("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle" / "access_token").exists():
        raise RuntimeError("Configure KAGGLE_API_TOKEN en .env o ~/.kaggle/access_token")
    RAW.mkdir(parents=True, exist_ok=True)
    for name, slug in DATASETS.items():
        destination = RAW / name
        destination.mkdir(exist_ok=True)
        kagglehub.dataset_download(slug, output_dir=str(destination))
    files = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(RAW.rglob("*")) if path.is_file()
    ]
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "datasets": DATASETS,
        "files": files,
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Descarga completa: {len(files)} archivos registrados")


if __name__ == "__main__":
    main()
