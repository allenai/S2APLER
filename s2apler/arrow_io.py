"""Arrow IPC conversion and loading for S2APLER paper data."""

from __future__ import annotations

import datetime
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import pyarrow as pa
import pyarrow.compute as pc

ARROW_BUNDLE_SCHEMA = "s2apler_arrow_bundle_v2"
PAPERS_FILE = "papers.arrow"
PAPER_AUTHORS_FILE = "paper_authors.arrow"
CLUSTERS_FILE = "clusters.arrow"
MANIFEST_FILE = "manifest.json"
BLOCK_INDEX_FILE = "block_index.json"
BLOCK_COUNTS_FILE = "block_counts.json"
PAPERS_BATCH_INDEX_FILE = "papers_batch_index.json"
PAPER_AUTHORS_BATCH_INDEX_FILE = "paper_authors_batch_index.json"
DEFAULT_PAPERS_BATCH_SIZE = 16_384
DEFAULT_PAPER_AUTHORS_BATCH_SIZE = 65_536
DEFAULT_CLUSTERS_BATCH_SIZE = 65_536

PAPERS_SCHEMA = pa.schema(
    [
        ("paper_id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("venue", pa.string()),
        ("journal_name", pa.string()),
        ("year", pa.int64()),
        ("corpus_paper_id", pa.int64()),
        ("doi", pa.string()),
        ("pmid", pa.string()),
        ("source_id", pa.string()),
        ("pdf_hash", pa.string()),
        ("source", pa.string()),
        ("block", pa.string()),
        ("source_uris", pa.list_(pa.string())),
    ]
)

PAPER_AUTHORS_SCHEMA = pa.schema(
    [
        ("paper_id", pa.string()),
        ("position", pa.int64()),
        ("first", pa.string()),
        ("middle", pa.list_(pa.string())),
        ("last", pa.string()),
        ("suffix", pa.string()),
        ("author_info_first", pa.string()),
        ("author_info_middle", pa.string()),
        ("author_info_last", pa.string()),
        ("author_info_suffix", pa.string()),
    ]
)

CLUSTERS_SCHEMA = pa.schema(
    [
        ("cluster_key", pa.string()),
        ("cluster_json", pa.string()),
    ]
)


def convert_json_dataset_to_arrow(
    *,
    papers_path: str | Path,
    output_dir: str | Path,
    clusters_path: str | Path | None = None,
    overwrite: bool = False,
    papers_batch_size: int = DEFAULT_PAPERS_BATCH_SIZE,
    paper_authors_batch_size: int = DEFAULT_PAPER_AUTHORS_BATCH_SIZE,
    clusters_batch_size: int = DEFAULT_CLUSTERS_BATCH_SIZE,
) -> Dict[str, Any]:
    """Convert S2APLER JSON artifacts into an indexed Arrow IPC bundle.

    Args:
        papers_path: Path to `papers.json`.
        output_dir: Directory where the Arrow bundle will be written.
        clusters_path: Optional path to `clusters.json`.
        overwrite: Whether to replace an existing S2APLER Arrow bundle.
        papers_batch_size: Maximum rows per papers IPC record batch.
        paper_authors_batch_size: Target maximum rows per author IPC record batch.
        clusters_batch_size: Maximum rows per cluster membership IPC record batch.

    Returns:
        A manifest dictionary describing written files, counts, and batch sizes.
    """
    papers_path = Path(papers_path)
    output_dir = Path(output_dir)
    clusters_path = None if clusters_path is None else Path(clusters_path)
    if papers_batch_size <= 0:
        raise ValueError("papers_batch_size must be positive")
    if paper_authors_batch_size <= 0:
        raise ValueError("paper_authors_batch_size must be positive")
    if clusters_batch_size <= 0:
        raise ValueError("clusters_batch_size must be positive")
    if not papers_path.exists():
        raise FileNotFoundError(f"Missing papers JSON: {papers_path}")
    if clusters_path is not None and not clusters_path.exists():
        raise FileNotFoundError(f"Missing clusters JSON: {clusters_path}")

    with papers_path.open(encoding="utf-8") as infile:
        papers = json.load(infile)
    if not isinstance(papers, dict):
        raise TypeError(f"Expected papers JSON object at {papers_path}")
    if not papers:
        raise ValueError(f"Papers JSON is empty: {papers_path}")

    _prepare_output_dir(output_dir, overwrite=overwrite)
    papers_batch_index: Dict[str, int] = {}
    paper_authors_batch_index: Dict[str, int] = {}
    block_index: DefaultDict[str, List[str]] = defaultdict(list)
    block_counts: Counter[str] = Counter()

    paper_count, author_count, paper_batch_count, author_batch_count = _write_papers_and_authors(
        papers=papers,
        output_dir=output_dir,
        papers_batch_index=papers_batch_index,
        paper_authors_batch_index=paper_authors_batch_index,
        block_index=block_index,
        block_counts=block_counts,
        papers_batch_size=papers_batch_size,
        paper_authors_batch_size=paper_authors_batch_size,
    )

    files: Dict[str, str] = {
        "papers": PAPERS_FILE,
        "paper_authors": PAPER_AUTHORS_FILE,
        "block_index": BLOCK_INDEX_FILE,
        "block_counts": BLOCK_COUNTS_FILE,
        "papers_batch_index": PAPERS_BATCH_INDEX_FILE,
        "paper_authors_batch_index": PAPER_AUTHORS_BATCH_INDEX_FILE,
    }
    cluster_membership_count = 0
    cluster_count = 0
    cluster_batch_count = 0
    if clusters_path is not None:
        cluster_count, cluster_membership_count, cluster_batch_count = _write_clusters(
            clusters_path=clusters_path,
            output_dir=output_dir,
            clusters_batch_size=clusters_batch_size,
        )
        files["clusters"] = CLUSTERS_FILE

    _write_json(output_dir / PAPERS_BATCH_INDEX_FILE, papers_batch_index)
    _write_json(output_dir / PAPER_AUTHORS_BATCH_INDEX_FILE, paper_authors_batch_index)
    _write_json(output_dir / BLOCK_INDEX_FILE, dict(block_index))
    _write_json(output_dir / BLOCK_COUNTS_FILE, dict(block_counts))
    largest_block, largest_block_size = block_counts.most_common(1)[0]

    manifest = {
        "schema": ARROW_BUNDLE_SCHEMA,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "papers_path": str(papers_path),
            "clusters_path": None if clusters_path is None else str(clusters_path),
        },
        "files": files,
        "counts": {
            "paper_count": int(paper_count),
            "paper_author_count": int(author_count),
            "block_count": int(len(block_counts)),
            "cluster_count": int(cluster_count),
            "cluster_membership_count": int(cluster_membership_count),
        },
        "largest_block": {
            "block": largest_block,
            "size": int(largest_block_size),
        },
        "batch_sizes": {
            "papers": int(papers_batch_size),
            "paper_authors": int(paper_authors_batch_size),
            "clusters": int(clusters_batch_size),
        },
        "batch_counts": {
            "papers": int(paper_batch_count),
            "paper_authors": int(author_batch_count),
            "clusters": int(cluster_batch_count),
        },
    }
    _write_json(output_dir / MANIFEST_FILE, manifest)
    return validate_arrow_bundle(output_dir)


def validate_arrow_bundle(arrow_dir: str | Path, *, require_clusters: bool = False) -> Dict[str, Any]:
    """Validate a S2APLER Arrow bundle enough for selective loading."""
    arrow_dir = normalize_arrow_bundle_dir(arrow_dir)
    manifest_path = arrow_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Arrow manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != ARROW_BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported Arrow bundle schema: {manifest.get('schema')!r}")

    required = [
        "papers",
        "paper_authors",
        "block_index",
        "block_counts",
        "papers_batch_index",
        "paper_authors_batch_index",
    ]
    if require_clusters:
        required.append("clusters")
    files = manifest.get("files", {})
    for key in required:
        if key not in files:
            raise ValueError(f"Arrow bundle manifest is missing file key {key!r}")
        path = arrow_dir / files[key]
        if not path.exists():
            raise FileNotFoundError(f"Arrow bundle file for {key!r} does not exist: {path}")

    _validate_ipc_schema(arrow_dir / files["papers"], PAPERS_SCHEMA, label="papers")
    _validate_ipc_schema(arrow_dir / files["paper_authors"], PAPER_AUTHORS_SCHEMA, label="paper_authors")
    if "clusters" in files:
        _validate_ipc_schema(arrow_dir / files["clusters"], CLUSTERS_SCHEMA, label="clusters")
    return manifest


def is_arrow_bundle_path(path: Any) -> bool:
    """Return whether `path` points to a S2APLER Arrow bundle directory or manifest."""
    if not isinstance(path, (str, Path)):
        return False
    path = Path(path)
    if path.is_dir():
        return (path / MANIFEST_FILE).exists()
    return path.name == MANIFEST_FILE and path.exists()


def normalize_arrow_bundle_dir(path: str | Path) -> Path:
    """Resolve a S2APLER Arrow bundle directory from a dir or manifest path."""
    path = Path(path)
    if path.is_file():
        if path.name != MANIFEST_FILE:
            raise ValueError(f"Arrow bundle file path must point to {MANIFEST_FILE}: {path}")
        return path.parent
    return path


def load_papers_from_arrow(
    arrow_dir: str | Path,
    *,
    paper_ids: Optional[Sequence[str]] = None,
    block_key: Optional[str] = None,
    max_block_size: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load raw S2APLER paper dictionaries from an Arrow bundle.

    Args:
        arrow_dir: Directory containing a S2APLER Arrow bundle.
        paper_ids: Optional explicit paper ids to load.
        block_key: Optional block key. `None` means load the full bundle. Empty
            string means select the largest block.
        max_block_size: Optional cap for block selection. Matching the JSON
            profiling path, capped block ids are sorted before loading.

    Returns:
        A `(papers, metadata)` tuple, where `papers` matches the input shape
        accepted by `PDData`.
    """
    arrow_dir = normalize_arrow_bundle_dir(arrow_dir)
    manifest = validate_arrow_bundle(arrow_dir)
    files = manifest["files"]
    if paper_ids is not None and block_key is not None:
        raise ValueError("paper_ids and block_key are mutually exclusive")
    if max_block_size < 0:
        raise ValueError("max_block_size cannot be negative")
    if max_block_size > 0 and block_key is None:
        raise ValueError("max_block_size requires block_key")

    selection_start = datetime.datetime.now(datetime.timezone.utc)
    selected_block: Optional[str] = None
    original_block_size: Optional[int] = None
    load_all = paper_ids is None and block_key is None
    if load_all:
        papers_table = _read_all_ipc_table(arrow_dir / files["papers"])
        authors_table = _read_all_ipc_table(arrow_dir / files["paper_authors"])
        selected_ids = [str(paper_id) for paper_id in papers_table["paper_id"].to_pylist()]
        original_block_size = len(selected_ids)
        papers = _papers_from_tables(selected_ids, papers_table, authors_table)
    elif paper_ids is None:
        block_index = _read_json(arrow_dir / files["block_index"])
        if not block_index:
            raise ValueError("Arrow bundle contains no blocks")
        selected_block = block_key or _largest_block_from_manifest(manifest, arrow_dir / files["block_counts"])
        if selected_block not in block_index:
            raise ValueError(f"Block {selected_block!r} not found in Arrow block index")
        selected_ids = [str(paper_id) for paper_id in block_index[selected_block]]
        original_block_size = len(selected_ids)
        if max_block_size > 0 and len(selected_ids) > max_block_size:
            selected_ids = sorted(selected_ids)[:max_block_size]
    else:
        selected_ids = [str(paper_id) for paper_id in paper_ids]
        original_block_size = len(selected_ids)

    if not load_all:
        papers_batch_index = _read_json(arrow_dir / files["papers_batch_index"])
        paper_authors_batch_index = _read_json(arrow_dir / files["paper_authors_batch_index"])
        papers_table = _read_indexed_rows(
            arrow_dir / files["papers"],
            batch_ids=_batch_ids_for_papers(selected_ids, papers_batch_index),
            key_column="paper_id",
            selected_ids=selected_ids,
        )
        authors_table = _read_indexed_rows(
            arrow_dir / files["paper_authors"],
            batch_ids=_batch_ids_for_papers(selected_ids, paper_authors_batch_index, missing_ok=True),
            key_column="paper_id",
            selected_ids=selected_ids,
        )
        papers = _papers_from_tables(selected_ids, papers_table, authors_table)

    metadata = {
        "arrow_dir": str(arrow_dir),
        "selected_block": selected_block,
        "original_block_size": int(original_block_size or 0),
        "load_all": bool(load_all),
        "loaded_paper_count": int(len(papers)),
        "requested_paper_count": int(len(selected_ids)),
        "selection_started_at_utc": selection_start.replace(microsecond=0).isoformat(),
        "manifest_counts": manifest.get("counts", {}),
    }
    return papers, metadata


def load_clusters_from_arrow(
    arrow_dir: str | Path,
    *,
    require_clusters: bool = False,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load `clusters.json`-shaped data from an Arrow bundle, if present."""
    arrow_dir = normalize_arrow_bundle_dir(arrow_dir)
    manifest = validate_arrow_bundle(arrow_dir, require_clusters=require_clusters)
    clusters_file = manifest.get("files", {}).get("clusters")
    if clusters_file is None:
        return None
    table = _read_all_ipc_table(arrow_dir / clusters_file)
    clusters: Dict[str, Dict[str, Any]] = {}
    for row in table.to_pylist():
        clusters[row["cluster_key"]] = json.loads(row["cluster_json"])
    return clusters


def load_pddata_from_arrow(
    arrow_dir: str | Path,
    *,
    name: str,
    mode: str = "inference",
    clusters: bool = False,
    n_jobs: int = 1,
    balanced_pair_sample: bool = False,
    **pddata_kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Build `PDData` from a complete Arrow bundle."""
    from s2apler.data import PDData

    papers, metadata = load_papers_from_arrow(arrow_dir)
    clusters_payload = load_clusters_from_arrow(arrow_dir, require_clusters=True) if clusters else None
    dataset = PDData(
        papers=papers,
        clusters=clusters_payload,
        name=name,
        mode=mode,
        n_jobs=n_jobs,
        balanced_pair_sample=balanced_pair_sample,
        **pddata_kwargs,
    )
    return dataset, metadata


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_file_names = (
        PAPERS_FILE,
        PAPER_AUTHORS_FILE,
        CLUSTERS_FILE,
        MANIFEST_FILE,
        BLOCK_INDEX_FILE,
        BLOCK_COUNTS_FILE,
        PAPERS_BATCH_INDEX_FILE,
        PAPER_AUTHORS_BATCH_INDEX_FILE,
    )
    existing_bundle_files = [file_name for file_name in bundle_file_names if (output_dir / file_name).exists()]
    if existing_bundle_files and not overwrite:
        raise FileExistsError(
            f"Arrow bundle files already exist at {output_dir}: {existing_bundle_files}; pass overwrite=True to replace them"
        )
    if overwrite:
        for file_name in bundle_file_names:
            path = output_dir / file_name
            if path.exists():
                path.unlink()


def _write_papers_and_authors(
    *,
    papers: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    papers_batch_index: Dict[str, int],
    paper_authors_batch_index: Dict[str, int],
    block_index: DefaultDict[str, List[str]],
    block_counts: Counter[str],
    papers_batch_size: int,
    paper_authors_batch_size: int,
) -> Tuple[int, int, int, int]:
    paper_rows: List[Dict[str, Any]] = []
    author_rows: List[Dict[str, Any]] = []
    paper_batch_count = 0
    author_batch_count = 0
    paper_count = 0
    author_count = 0

    with pa.OSFile(str(output_dir / PAPERS_FILE), "wb") as papers_sink:
        with pa.OSFile(str(output_dir / PAPER_AUTHORS_FILE), "wb") as authors_sink:
            with pa.ipc.new_file(papers_sink, PAPERS_SCHEMA) as papers_writer:
                with pa.ipc.new_file(authors_sink, PAPER_AUTHORS_SCHEMA) as authors_writer:
                    for raw_paper_id, paper in papers.items():
                        paper_id = str(raw_paper_id)
                        row = _paper_row(paper_id, paper)
                        paper_rows.append(row)
                        block = row["block"] or ""
                        block_index[block].append(paper_id)
                        block_counts[block] += 1
                        paper_count += 1

                        authors = list(_author_rows(paper_id, paper.get("authors", [])))
                        if author_rows and len(author_rows) + len(authors) > paper_authors_batch_size:
                            author_batch_count += _write_record_batch(
                                authors_writer,
                                PAPER_AUTHORS_SCHEMA,
                                author_rows,
                                paper_authors_batch_index,
                                author_batch_count,
                            )
                            author_rows = []
                        author_rows.extend(authors)
                        author_count += len(authors)

                        if len(paper_rows) >= papers_batch_size:
                            paper_batch_count += _write_record_batch(
                                papers_writer,
                                PAPERS_SCHEMA,
                                paper_rows,
                                papers_batch_index,
                                paper_batch_count,
                            )
                            paper_rows = []
                        if len(author_rows) >= paper_authors_batch_size:
                            author_batch_count += _write_record_batch(
                                authors_writer,
                                PAPER_AUTHORS_SCHEMA,
                                author_rows,
                                paper_authors_batch_index,
                                author_batch_count,
                            )
                            author_rows = []

                    if paper_rows:
                        paper_batch_count += _write_record_batch(
                            papers_writer,
                            PAPERS_SCHEMA,
                            paper_rows,
                            papers_batch_index,
                            paper_batch_count,
                        )
                    if author_rows:
                        author_batch_count += _write_record_batch(
                            authors_writer,
                            PAPER_AUTHORS_SCHEMA,
                            author_rows,
                            paper_authors_batch_index,
                            author_batch_count,
                        )
    return paper_count, author_count, paper_batch_count, author_batch_count


def _write_clusters(
    *,
    clusters_path: Path,
    output_dir: Path,
    clusters_batch_size: int,
) -> Tuple[int, int, int]:
    with clusters_path.open(encoding="utf-8") as infile:
        clusters = json.load(infile)
    if not isinstance(clusters, dict):
        raise TypeError(f"Expected clusters JSON object at {clusters_path}")
    rows: List[Dict[str, Any]] = []
    cluster_count = len(clusters)
    membership_count = 0
    batch_count = 0
    with pa.OSFile(str(output_dir / CLUSTERS_FILE), "wb") as sink:
        with pa.ipc.new_file(sink, CLUSTERS_SCHEMA) as writer:
            for raw_cluster_id, cluster in clusters.items():
                if not isinstance(cluster, Mapping):
                    raise TypeError(f"Expected cluster object for cluster {raw_cluster_id!r}")
                sourced_paper_ids = cluster.get("sourced_paper_ids", [])
                if not isinstance(sourced_paper_ids, list):
                    raise TypeError(f"Expected sourced_paper_ids list for cluster {raw_cluster_id!r}")
                rows.append(
                    {
                        "cluster_key": str(raw_cluster_id),
                        "cluster_json": json.dumps(cluster, separators=(",", ":"), sort_keys=True),
                    }
                )
                membership_count += len(sourced_paper_ids)
                if len(rows) >= clusters_batch_size:
                    _write_batch(writer, CLUSTERS_SCHEMA, rows)
                    rows = []
                    batch_count += 1
            if rows:
                _write_batch(writer, CLUSTERS_SCHEMA, rows)
                batch_count += 1
    return cluster_count, membership_count, batch_count


def _paper_row(paper_id: str, paper: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": _optional_string(paper.get("title", "")),
        "abstract": _optional_string(paper.get("abstract")),
        "venue": _optional_string(paper.get("venue")),
        "journal_name": _optional_string(paper.get("journal_name")),
        "year": _optional_int(paper.get("year")),
        "corpus_paper_id": _optional_int(paper.get("corpus_paper_id")),
        "doi": _optional_string(paper.get("doi")),
        "pmid": _optional_string(paper.get("pmid")),
        "source_id": _optional_string(paper.get("source_id")),
        "pdf_hash": _optional_string(paper.get("pdf_hash")),
        "source": _optional_string(paper.get("source")),
        "block": _optional_string(paper.get("block")),
        "source_uris": _optional_string_list(paper.get("source_uris")),
    }


def _author_rows(paper_id: str, authors: Any) -> Iterable[Dict[str, Any]]:
    if authors is None:
        return
    if not isinstance(authors, list):
        raise TypeError(f"Expected authors list for paper {paper_id!r}, found {type(authors).__name__}")
    for fallback_position, author in enumerate(authors):
        if not isinstance(author, Mapping):
            raise TypeError(f"Expected author object for paper {paper_id!r}")
        yield {
            "paper_id": paper_id,
            "position": _optional_int(author.get("position"), fallback=fallback_position),
            "first": _optional_string(author.get("first")),
            "middle": _optional_string_list(author.get("middle", [])) or [],
            "last": _optional_string(author.get("last")),
            "suffix": _optional_string(author.get("suffix")),
            "author_info_first": _optional_string(author.get("author_info_first")),
            "author_info_middle": _optional_string(author.get("author_info_middle")),
            "author_info_last": _optional_string(author.get("author_info_last")),
            "author_info_suffix": _optional_string(author.get("author_info_suffix")),
        }


def _write_record_batch(
    writer: pa.ipc.RecordBatchFileWriter,
    schema: pa.Schema,
    rows: Sequence[Mapping[str, Any]],
    batch_index: Dict[str, int],
    batch_number: int,
) -> int:
    for row in rows:
        paper_id = row["paper_id"]
        batch_index.setdefault(str(paper_id), batch_number)
    _write_batch(writer, schema, rows)
    return 1


def _write_batch(
    writer: pa.ipc.RecordBatchFileWriter,
    schema: pa.Schema,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    arrays = [pa.array([row[field.name] for row in rows], type=field.type) for field in schema]
    writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))


def _read_indexed_rows(
    path: Path,
    *,
    batch_ids: Sequence[int],
    key_column: str,
    selected_ids: Sequence[str],
) -> pa.Table:
    if not batch_ids:
        return pa.Table.from_batches([], schema=_schema_for_path(path))
    batches = []
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        for batch_id in sorted(set(batch_ids)):
            if batch_id < 0 or batch_id >= reader.num_record_batches:
                raise IndexError(f"Batch index {batch_id} is outside {path}")
            batches.append(reader.get_batch(batch_id))
    table = pa.Table.from_batches(batches)
    selected_values = pa.array([str(value) for value in selected_ids], type=pa.string())
    return table.filter(pc.is_in(table[key_column], value_set=selected_values))


def _read_all_ipc_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        return reader.read_all()


def _schema_for_path(path: Path) -> pa.Schema:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).schema


def _papers_from_tables(
    selected_ids: Sequence[str],
    papers_table: pa.Table,
    authors_table: pa.Table,
) -> Dict[str, Dict[str, Any]]:
    paper_rows = {row["paper_id"]: row for row in papers_table.to_pylist()}
    author_rows_by_paper: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in authors_table.to_pylist():
        author_rows_by_paper[row["paper_id"]].append(row)

    output: Dict[str, Dict[str, Any]] = {}
    missing = [paper_id for paper_id in selected_ids if paper_id not in paper_rows]
    if missing:
        raise ValueError(f"Arrow bundle is missing selected paper ids: {missing[:10]}")

    for paper_id in selected_ids:
        row = paper_rows[paper_id]
        output[paper_id] = {
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "authors": [_author_dict(author_row) for author_row in author_rows_by_paper.get(paper_id, [])],
            "venue": row.get("venue"),
            "journal_name": row.get("journal_name"),
            "year": row.get("year"),
            "doi": row.get("doi"),
            "pmid": row.get("pmid"),
            "source": row.get("source"),
            "source_id": row.get("source_id"),
            "pdf_hash": row.get("pdf_hash"),
            "block": row.get("block"),
            "corpus_paper_id": row.get("corpus_paper_id"),
            "source_uris": row.get("source_uris"),
        }
    return output


def _author_dict(row: Mapping[str, Any]) -> Dict[str, Any]:
    author = {
        "position": row.get("position"),
        "first": row.get("first"),
        "middle": row.get("middle") or [],
        "last": row.get("last"),
        "suffix": row.get("suffix"),
    }
    for key in (
        "author_info_first",
        "author_info_middle",
        "author_info_last",
        "author_info_suffix",
    ):
        if row.get(key) is not None:
            author[key] = row[key]
    return author


def _batch_ids_for_papers(
    paper_ids: Sequence[str],
    batch_index: Mapping[str, Any],
    *,
    missing_ok: bool = False,
) -> List[int]:
    batch_ids: List[int] = []
    missing: List[str] = []
    for paper_id in paper_ids:
        value = batch_index.get(str(paper_id))
        if value is None:
            if not missing_ok:
                missing.append(str(paper_id))
            continue
        if isinstance(value, list):
            batch_ids.extend(int(batch_id) for batch_id in value)
        else:
            batch_ids.append(int(value))
    if missing:
        raise ValueError(f"Arrow batch index is missing paper ids: {missing[:10]}")
    return batch_ids


def _largest_block_from_counts(path: Path) -> str:
    counts = _read_json(path)
    if not counts:
        raise ValueError("Arrow bundle contains no block counts")
    return max(counts.items(), key=lambda item: int(item[1]))[0]


def _largest_block_from_manifest(manifest: Mapping[str, Any], counts_path: Path) -> str:
    largest_block = manifest.get("largest_block", {})
    block = largest_block.get("block")
    if block is not None:
        return str(block)
    return _largest_block_from_counts(counts_path)


def _validate_ipc_schema(path: Path, expected_schema: pa.Schema, *, label: str) -> None:
    with pa.memory_map(str(path), "r") as source:
        actual_schema = pa.ipc.open_file(source).schema
    if actual_schema != expected_schema:
        raise ValueError(f"Unexpected {label} schema in {path}: {actual_schema!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _optional_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _optional_int(value: Any, *, fallback: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return fallback
    return int(value)
