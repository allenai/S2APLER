from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

from s2apler.arrow_io import convert_json_dataset_to_arrow

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert S2APLER JSON data into an indexed Arrow IPC bundle.")
    parser.add_argument("--papers", default="data/papers.json", help="Path to papers.json.")
    parser.add_argument(
        "--clusters",
        default=None,
        help="Optional path to clusters.json. Defaults to data/clusters.json when present; pass an empty string to omit.",
    )
    parser.add_argument("--output-dir", required=True, help="Output Arrow bundle directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing S2APLER Arrow bundle.")
    parser.add_argument("--papers-batch-size", type=int, default=16_384)
    parser.add_argument("--paper-authors-batch-size", type=int, default=65_536)
    parser.add_argument("--clusters-batch-size", type=int, default=65_536)
    parser.add_argument("--write-json", default="", help="Optional path for the manifest summary JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    papers_path = _resolve_path(args.papers)
    if args.clusters is None:
        default_clusters_path = _resolve_path("data/clusters.json")
        clusters_path = default_clusters_path if default_clusters_path.exists() else None
    else:
        clusters_path = _resolve_path(args.clusters) if args.clusters else None
    output_dir = _resolve_path(args.output_dir)

    start = time.perf_counter()
    manifest = convert_json_dataset_to_arrow(
        papers_path=papers_path,
        clusters_path=clusters_path,
        output_dir=output_dir,
        overwrite=args.overwrite,
        papers_batch_size=args.papers_batch_size,
        paper_authors_batch_size=args.paper_authors_batch_size,
        clusters_batch_size=args.clusters_batch_size,
    )
    result = {
        "papers_path": str(papers_path),
        "clusters_path": None if clusters_path is None else str(clusters_path),
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.perf_counter() - start, 6),
        "manifest": manifest,
    }
    if args.write_json:
        write_json_path = _resolve_path(args.write_json)
        write_json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
