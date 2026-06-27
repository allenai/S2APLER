from __future__ import annotations

import argparse
import cProfile
import contextlib
import datetime
import gc
import io
import json
import os
import pickle
import platform
import pstats
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_dataset"
DEFAULT_MODEL_PATH = DATA_DIR / "prod_model.pickle"
RESULT_JSON_START = "===S2APLER_PROFILE_RESULT_START==="
RESULT_JSON_END = "===S2APLER_PROFILE_RESULT_END==="
DEFAULT_ENV_KEYS = ("PYTHONHASHSEED", "OMP_NUM_THREADS", "S2APLER_CACHE")
SYNTHETIC_BLOCK_KEY = "synthetic_large_block"

try:
    import psutil
except ModuleNotFoundError:
    psutil = None  # type: ignore[assignment]


@dataclass
class TimedCallStats:
    seconds: float = 0.0
    calls: int = 0


class ProcessTreeRSSMonitor:
    """Monitor best-effort peak RSS for this process and child workers."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process() if psutil is not None else None

    def _read_rss_bytes(self) -> int:
        if psutil is None or self._process is None:
            return _fallback_rss_bytes()

        rss_total = 0
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except psutil.Error:
            pass
        for proc in processes:
            try:
                rss_total += int(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return rss_total

    def sample_rss_bytes(self) -> int:
        rss = self._read_rss_bytes()
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        return rss

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_rss_bytes()
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "ProcessTreeRSSMonitor":
        self.peak_rss_bytes = self.sample_rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_gb(self) -> float:
        return self.peak_rss_bytes / (1024**3)

    @property
    def rss_scope(self) -> str:
        return "process_tree" if psutil is not None else "current_process"


def _fallback_rss_bytes() -> int:
    """Return current-process RSS without adding a runtime dependency."""
    if sys.platform.startswith("win"):
        return _windows_current_process_rss_bytes()
    try:
        import resource

        ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform.startswith("darwin"):
            return ru_maxrss
        return ru_maxrss * 1024
    except Exception:
        return 0


def _windows_current_process_rss_bytes() -> int:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if not ok:
            return 0
        return int(counters.WorkingSetSize)
    except Exception:
        return 0


@contextlib.contextmanager
def timed_call(target: Any, attr_name: str) -> Iterator[TimedCallStats]:
    """Temporarily wrap target.attr_name and collect elapsed time/call count."""
    original = getattr(target, attr_name)
    if not callable(original):
        raise TypeError(f"{target!r}.{attr_name} is not callable")

    stats = TimedCallStats()

    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stats.seconds += time.perf_counter() - start
            stats.calls += 1

    setattr(target, attr_name, _wrapper)
    try:
        yield stats
    finally:
        setattr(target, attr_name, original)


def _run_git(args: List[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_run_metadata(script_path: Path, argv: Optional[List[str]] = None) -> Dict[str, Any]:
    git_status = _run_git(["status", "--porcelain"])
    env_snapshot = {key: os.environ[key] for key in DEFAULT_ENV_KEYS if key in os.environ}
    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "script": str(script_path.resolve()),
        "argv": list(sys.argv if argv is None else argv),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": None if git_status is None else bool(git_status),
        "env": env_snapshot,
    }


def extract_marked_json_payload(stdout_text: str) -> Dict[str, Any]:
    start = stdout_text.find(RESULT_JSON_START)
    end = stdout_text.find(RESULT_JSON_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Failed to parse result JSON markers from subprocess output")
    return json.loads(stdout_text[start + len(RESULT_JSON_START) : end].strip())


def _write_profile_output(
    profiler: cProfile.Profile, output_path: Path, elapsed_seconds: float
) -> List[Dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(80)
    output_path.write_text(
        stats_stream.getvalue() + f"\nTotal profiled runtime: {elapsed_seconds:.6f}s\n",
        encoding="utf-8",
    )

    top_functions: List[Dict[str, Any]] = []
    for func, stat in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:20]:
        primitive_calls, total_calls, total_time, cumulative_time, _callers = stat
        filename, line_no, func_name = func
        top_functions.append(
            {
                "function": f"{Path(filename).name}:{line_no}:{func_name}",
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "total_time_seconds": round(float(total_time), 6),
                "cumulative_time_seconds": round(float(cumulative_time), 6),
            }
        )
    return top_functions


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _default_data_root() -> Path:
    if (DATA_DIR / "papers.json").exists():
        return DATA_DIR
    return TEST_DATA_DIR


def _data_paths(data_root: Path) -> Dict[str, Optional[Path]]:
    paths: Dict[str, Optional[Path]] = {
        "papers": data_root / "papers.json",
        "clusters": data_root / "clusters.json",
        "cluster_seeds": data_root / "cluster_seeds.json",
    }
    return {key: path if path is not None and path.exists() else None for key, path in paths.items()}


def _is_arrow_data_root(data_root: Path) -> bool:
    from s2apler.arrow_io import is_arrow_bundle_path

    return is_arrow_bundle_path(data_root)


def _load_arrow_manifest(data_root: Path) -> Dict[str, Any]:
    from s2apler.arrow_io import normalize_arrow_bundle_dir, validate_arrow_bundle

    return validate_arrow_bundle(normalize_arrow_bundle_dir(data_root))


def _load_arrow_block_counts(data_root: Path) -> Dict[str, int]:
    from s2apler.arrow_io import normalize_arrow_bundle_dir

    arrow_root = normalize_arrow_bundle_dir(data_root)
    counts = _load_json(arrow_root / "block_counts.json")
    return {str(block): int(size) for block, size in counts.items()}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_papers_from_root(data_root: Path) -> Dict[str, Dict[str, Any]]:
    paths = _data_paths(data_root)
    papers_path = paths["papers"]
    if papers_path is None:
        raise FileNotFoundError(f"Missing papers.json under {data_root}")
    return _load_json(papers_path)


def _select_largest_block(papers: Dict[str, Dict[str, Any]]) -> Tuple[str, int]:
    block_counts: Counter[str] = Counter()
    for paper in papers.values():
        block = str(paper.get("block") or "")
        block_counts[block] += 1
    if not block_counts:
        raise ValueError("No papers available for block scan")
    return block_counts.most_common(1)[0]


def _select_block_from_papers(
    papers: Dict[str, Dict[str, Any]], block_key: str, max_block_size: int
) -> Tuple[str, List[str], int]:
    blocks: Dict[str, List[str]] = {}
    for paper_id, paper in papers.items():
        blocks.setdefault(str(paper.get("block") or ""), []).append(paper_id)
    if not blocks:
        raise ValueError("No papers available for block selection")

    selected_block = block_key
    if not selected_block:
        selected_block = max(blocks.items(), key=lambda item: len(item[1]))[0]
    if selected_block not in blocks:
        raise ValueError(f"Block {selected_block!r} not found. Available examples: {sorted(blocks)[:10]}")

    paper_ids = list(blocks[selected_block])
    original_size = len(paper_ids)
    if max_block_size > 0 and len(paper_ids) > max_block_size:
        paper_ids = sorted(paper_ids)[:max_block_size]
    return selected_block, paper_ids, original_size


def _block_summary(papers: Dict[str, Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    block_counts: Counter[str] = Counter()
    for paper in papers.values():
        block_counts[str(paper.get("block") or "")] += 1
    return [
        {"block": block, "size": int(size), "num_pairs": int(size * (size - 1) // 2)}
        for block, size in block_counts.most_common(top_k)
    ]


def _make_synthetic_papers(
    source_papers: Dict[str, Dict[str, Any]],
    block_size: int,
    source_block: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    if block_size <= 1:
        raise ValueError("--block-size must be greater than 1")

    if source_block is None:
        source_block, _ = _select_largest_block(source_papers)

    templates = [paper for paper in source_papers.values() if str(paper.get("block") or "") == source_block]
    if not templates:
        raise ValueError(f"Source block {source_block!r} was not found")

    synthetic: Dict[str, Dict[str, Any]] = {}
    for index in range(block_size):
        template = dict(templates[index % len(templates)])
        paper_id = f"synthetic_{index:07d}"
        template["block"] = SYNTHETIC_BLOCK_KEY
        template["sourced_paper_id"] = paper_id
        template["corpus_paper_id"] = None
        template["doi"] = None
        template["pmid"] = None
        template["pdf_hash"] = None
        template["source_id"] = None
        template["source_uris"] = None
        synthetic[paper_id] = template
    return synthetic


def _load_dataset(
    *,
    data_root: Path,
    n_jobs: int,
    synthetic_block_size: int = 0,
    synthetic_source_block: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    from s2apler.data import PDData

    if synthetic_block_size > 0:
        source_papers = _load_papers_from_root(data_root)
        papers = _make_synthetic_papers(
            source_papers,
            block_size=synthetic_block_size,
            source_block=synthetic_source_block,
        )
        start = time.perf_counter()
        dataset = PDData(
            papers=papers,
            name=f"synthetic_{synthetic_block_size}",
            mode="inference",
            n_jobs=n_jobs,
            balanced_pair_sample=False,
        )
        return dataset, {
            "synthetic": True,
            "source_data_root": str(data_root),
            "source_block": synthetic_source_block,
            "requested_block_size": int(synthetic_block_size),
            "dataset_build_seconds": round(time.perf_counter() - start, 6),
        }

    if _is_arrow_data_root(data_root):
        start = time.perf_counter()
        dataset = PDData(
            papers=str(data_root),
            name=data_root.name or "paper_clustering_dataset",
            mode="train",
            n_jobs=n_jobs,
            balanced_pair_sample=False,
        )
        return dataset, {
            "synthetic": False,
            "arrow": True,
            "data_root": str(data_root),
            "mode": dataset.mode,
            "dataset_build_seconds": round(time.perf_counter() - start, 6),
            "arrow_metadata": dataset.arrow_metadata,
        }

    paths = _data_paths(data_root)
    papers_path = paths["papers"]
    if papers_path is None:
        raise FileNotFoundError(f"Missing papers.json under {data_root}")
    clusters_path = paths["clusters"]
    cluster_seeds_path = paths["cluster_seeds"]
    mode = "train" if clusters_path is not None else "inference"
    start = time.perf_counter()
    dataset = PDData(
        papers=str(papers_path),
        clusters=str(clusters_path) if clusters_path is not None else None,
        cluster_seeds=str(cluster_seeds_path) if cluster_seeds_path is not None else None,
        name=data_root.name or "paper_clustering_dataset",
        mode=mode,
        n_jobs=n_jobs,
        balanced_pair_sample=False,
    )
    return dataset, {
        "synthetic": False,
        "data_root": str(data_root),
        "papers_path": str(papers_path),
        "clusters_path": None if clusters_path is None else str(clusters_path),
        "cluster_seeds_path": None if cluster_seeds_path is None else str(cluster_seeds_path),
        "mode": mode,
        "dataset_build_seconds": round(time.perf_counter() - start, 6),
    }


def _load_block_only_dataset(
    data_root: Path,
    n_jobs: int,
    block_key: str,
    max_block_size: int,
) -> Tuple[Any, Dict[str, Any], str, List[str], int]:
    from s2apler.data import PDData

    paths = _data_paths(data_root)
    papers_path = paths["papers"]
    if papers_path is None:
        raise FileNotFoundError(f"Missing papers.json under {data_root}")

    scan_start = time.perf_counter()
    all_papers = _load_json(papers_path)
    selected_block, paper_ids, original_size = _select_block_from_papers(all_papers, block_key, max_block_size)
    selected_papers = {paper_id: all_papers[paper_id] for paper_id in paper_ids}
    paper_count = len(all_papers)
    scan_seconds = time.perf_counter() - scan_start
    del all_papers
    gc.collect()

    build_start = time.perf_counter()
    dataset = PDData(
        papers=selected_papers,
        name=f"paper_clustering_{selected_block}",
        mode="inference",
        n_jobs=n_jobs,
        balanced_pair_sample=False,
    )
    build_seconds = time.perf_counter() - build_start
    return (
        dataset,
        {
            "synthetic": False,
            "block_only": True,
            "data_root": str(data_root),
            "papers_path": str(papers_path),
            "clusters_path": None,
            "cluster_seeds_path": None,
            "mode": "inference",
            "paper_count": int(paper_count),
            "scan_seconds": round(scan_seconds, 6),
            "dataset_build_seconds": round(build_seconds, 6),
        },
        selected_block,
        paper_ids,
        original_size,
    )


def _load_arrow_block_only_dataset(
    arrow_root: Path,
    n_jobs: int,
    block_key: str,
    max_block_size: int,
) -> Tuple[Any, Dict[str, Any], str, List[str], int]:
    from s2apler.arrow_io import load_papers_from_arrow
    from s2apler.data import PDData

    load_start = time.perf_counter()
    selected_papers, arrow_metadata = load_papers_from_arrow(
        arrow_root,
        block_key=block_key,
        max_block_size=max_block_size,
    )
    arrow_load_seconds = time.perf_counter() - load_start
    selected_block = arrow_metadata["selected_block"]
    if selected_block is None:
        raise ValueError("Arrow block loading did not return a selected block")
    paper_ids = list(selected_papers.keys())
    original_size = int(arrow_metadata["original_block_size"])

    build_start = time.perf_counter()
    dataset = PDData(
        papers=selected_papers,
        name=f"paper_clustering_arrow_{selected_block}",
        mode="inference",
        n_jobs=n_jobs,
        balanced_pair_sample=False,
    )
    build_seconds = time.perf_counter() - build_start
    return (
        dataset,
        {
            "synthetic": False,
            "block_only": True,
            "arrow": True,
            "arrow_root": str(arrow_root),
            "mode": "inference",
            "arrow_load_seconds": round(arrow_load_seconds, 6),
            "dataset_build_seconds": round(build_seconds, 6),
            "arrow_metadata": arrow_metadata,
        },
        selected_block,
        paper_ids,
        original_size,
    )


def _select_block(dataset: Any, block_key: str, max_block_size: int) -> Tuple[str, List[str], int]:
    blocks = dataset.get_blocks()
    if not blocks:
        raise ValueError("Dataset contains no blocks")

    selected_block = block_key
    if not selected_block:
        selected_block = max(blocks.items(), key=lambda item: len(item[1]))[0]
    if selected_block not in blocks:
        raise ValueError(f"Block {selected_block!r} not found. Available examples: {sorted(blocks)[:10]}")

    papers = list(blocks[selected_block])
    original_size = len(papers)
    if max_block_size > 0 and len(papers) > max_block_size:
        papers = sorted(papers)[:max_block_size]
    return selected_block, papers, original_size


def _load_clusterer(model_path: Path, n_jobs: int, batch_size: int) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model pickle: {model_path}")
    with model_path.open("rb") as infile:
        loaded = pickle.load(infile)
    clusterer = loaded["clusterer"] if isinstance(loaded, dict) and "clusterer" in loaded else loaded
    clusterer.n_jobs = n_jobs
    clusterer.use_cache = False
    clusterer.batch_size = batch_size
    for classifier_name in ("classifier", "nameless_classifier"):
        classifier = getattr(clusterer, classifier_name, None)
        if classifier is not None and hasattr(classifier, "set_params"):
            try:
                classifier.set_params(n_jobs=n_jobs)
            except ValueError:
                pass
    return clusterer


def _timed_prediction(
    clusterer: Any, block: Dict[str, List[str]], dataset: Any
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    import s2apler.model as model_module

    timed_sections: Dict[str, Any] = {}
    with contextlib.ExitStack() as stack:
        featurize_stats = stack.enter_context(timed_call(model_module, "many_pairs_featurize"))
        classifier = getattr(clusterer, "classifier", None)
        classifier_stats = (
            stack.enter_context(timed_call(classifier, "predict_proba"))
            if classifier is not None and hasattr(classifier, "predict_proba")
            else TimedCallStats()
        )
        nameless_classifier = getattr(clusterer, "nameless_classifier", None)
        nameless_stats = (
            stack.enter_context(timed_call(nameless_classifier, "predict_proba"))
            if nameless_classifier is not None and hasattr(nameless_classifier, "predict_proba")
            else TimedCallStats()
        )
        predictions, _dists = clusterer.predict(
            block,
            dataset,
            dists=None,
            cluster_model_params=None,
            partial_supervision={},
            use_s2_clusters=False,
        )

    timed_sections["many_pairs_featurize"] = {
        "seconds": round(featurize_stats.seconds, 6),
        "calls": int(featurize_stats.calls),
    }
    timed_sections["classifier_predict_proba"] = {
        "seconds": round(classifier_stats.seconds, 6),
        "calls": int(classifier_stats.calls),
    }
    timed_sections["nameless_classifier_predict_proba"] = {
        "seconds": round(nameless_stats.seconds, 6),
        "calls": int(nameless_stats.calls),
    }
    return predictions, timed_sections


def _cluster_digest(clusters: Dict[str, List[str]]) -> Dict[str, Any]:
    sizes = sorted((len(values) for values in clusters.values()), reverse=True)
    return {
        "num_clusters": int(len(sizes)),
        "assigned_papers": int(sum(sizes)),
        "cluster_sizes_top10": [int(size) for size in sizes[:10]],
    }


def _make_capped_pairs(papers: Sequence[str], max_pairs: int) -> List[Tuple[str, str, float]]:
    if max_pairs < 0:
        raise ValueError("--max-pairs cannot be negative")

    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            pairs.append((papers[i], papers[j], np.nan))
            if max_pairs > 0 and len(pairs) >= max_pairs:
                return pairs
    return pairs


def run_largest_block(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = _resolve_path(args.data_root) if args.data_root else _default_data_root()
    model_path = _resolve_path(args.model_path)
    profile_output_path = _resolve_path(args.profile_output_path)
    arrow_root = (
        _resolve_path(args.arrow_root) if args.arrow_root else data_root if _is_arrow_data_root(data_root) else None
    )

    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as monitor:
        if arrow_root is not None:
            dataset, dataset_metadata, block_key, papers, original_block_size = _load_arrow_block_only_dataset(
                arrow_root=arrow_root,
                n_jobs=args.n_jobs,
                block_key=args.block,
                max_block_size=args.max_block_size,
            )
        elif args.block_only:
            dataset, dataset_metadata, block_key, papers, original_block_size = _load_block_only_dataset(
                data_root=data_root,
                n_jobs=args.n_jobs,
                block_key=args.block,
                max_block_size=args.max_block_size,
            )
        else:
            dataset, dataset_metadata = _load_dataset(data_root=data_root, n_jobs=args.n_jobs)
            block_key, papers, original_block_size = _select_block(dataset, args.block, args.max_block_size)
        clusterer = _load_clusterer(model_path=model_path, n_jobs=args.n_jobs, batch_size=args.batch_size)
        block = {block_key: papers}
        profiler = cProfile.Profile()
        predict_start = time.perf_counter()
        profiler.enable()
        pred_clusters, timed_sections = _timed_prediction(clusterer, block, dataset)
        profiler.disable()
        predict_seconds = time.perf_counter() - predict_start

    total_seconds = time.perf_counter() - total_start
    top_functions = _write_profile_output(profiler, profile_output_path, predict_seconds)
    block_size = len(papers)
    return {
        "command": "largest-block",
        "data": dataset_metadata,
        "model_path": str(model_path),
        "block": block_key,
        "original_block_size": int(original_block_size),
        "effective_block_size": int(block_size),
        "num_pairs": int(block_size * (block_size - 1) // 2),
        "n_jobs": int(args.n_jobs),
        "batch_size": int(args.batch_size),
        "dataset_build_seconds": dataset_metadata["dataset_build_seconds"],
        "predict_seconds": round(predict_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "timed_sections": timed_sections,
        "peak_rss_gb": round(monitor.peak_gb, 6),
        "rss_scope": monitor.rss_scope,
        "clusters": _cluster_digest(pred_clusters),
        "profile_output_path": str(profile_output_path),
        "top_cumulative_functions": top_functions,
        "run_metadata": build_run_metadata(Path(__file__).resolve()),
    }


def run_synthetic_large_block(args: argparse.Namespace) -> Dict[str, Any]:
    if args.block_size <= 1:
        raise ValueError("--block-size is required and must be greater than 1")
    data_root = _resolve_path(args.source_data_root) if args.source_data_root else _default_data_root()
    model_path = _resolve_path(args.model_path)
    profile_output_path = _resolve_path(args.profile_output_path)

    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as monitor:
        dataset, dataset_metadata = _load_dataset(
            data_root=data_root,
            n_jobs=args.n_jobs,
            synthetic_block_size=args.block_size,
            synthetic_source_block=args.source_block or None,
        )
        block_key, papers, original_block_size = _select_block(dataset, SYNTHETIC_BLOCK_KEY, args.max_block_size)
        clusterer = _load_clusterer(model_path=model_path, n_jobs=args.n_jobs, batch_size=args.batch_size)
        block = {block_key: papers}
        profiler = cProfile.Profile()
        predict_start = time.perf_counter()
        profiler.enable()
        pred_clusters, timed_sections = _timed_prediction(clusterer, block, dataset)
        profiler.disable()
        predict_seconds = time.perf_counter() - predict_start

    total_seconds = time.perf_counter() - total_start
    top_functions = _write_profile_output(profiler, profile_output_path, predict_seconds)
    block_size = len(papers)
    return {
        "command": "synthetic-large-block",
        "data": dataset_metadata,
        "model_path": str(model_path),
        "block": block_key,
        "original_block_size": int(original_block_size),
        "effective_block_size": int(block_size),
        "num_pairs": int(block_size * (block_size - 1) // 2),
        "n_jobs": int(args.n_jobs),
        "batch_size": int(args.batch_size),
        "dataset_build_seconds": dataset_metadata["dataset_build_seconds"],
        "predict_seconds": round(predict_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "timed_sections": timed_sections,
        "peak_rss_gb": round(monitor.peak_gb, 6),
        "rss_scope": monitor.rss_scope,
        "clusters": _cluster_digest(pred_clusters),
        "profile_output_path": str(profile_output_path),
        "top_cumulative_functions": top_functions,
        "run_metadata": build_run_metadata(Path(__file__).resolve()),
    }


def run_pair_features(args: argparse.Namespace) -> Dict[str, Any]:
    from s2apler.featurizer import FeaturizationInfo, many_pairs_featurize

    if args.synthetic_block_size > 0 and args.synthetic_block_size <= 1:
        raise ValueError("--synthetic-block-size must be greater than 1")

    data_root = _resolve_path(args.data_root) if args.data_root else _default_data_root()
    profile_output_path = _resolve_path(args.profile_output_path)
    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as monitor:
        if args.synthetic_block_size <= 0 and _is_arrow_data_root(data_root):
            dataset, dataset_metadata, block_key, papers, original_block_size = _load_arrow_block_only_dataset(
                arrow_root=data_root,
                n_jobs=args.n_jobs,
                block_key=args.block,
                max_block_size=args.max_block_size,
            )
        else:
            dataset, dataset_metadata = _load_dataset(
                data_root=data_root,
                n_jobs=args.n_jobs,
                synthetic_block_size=args.synthetic_block_size,
                synthetic_source_block=args.source_block or None,
            )
            selected_block = SYNTHETIC_BLOCK_KEY if args.synthetic_block_size > 0 else args.block
            block_key, papers, original_block_size = _select_block(dataset, selected_block, args.max_block_size)
        all_pairs = _make_capped_pairs(papers, args.max_pairs)
        featurization_info = FeaturizationInfo()
        profiler = cProfile.Profile()
        featurize_start = time.perf_counter()
        profiler.enable()
        features, labels, _nameless = many_pairs_featurize(
            all_pairs,
            dataset,
            featurization_info,
            args.n_jobs,
            use_cache=False,
            chunk_size=args.chunk_size,
            nameless_featurizer_info=None,
            nan_value=np.nan,
        )
        profiler.disable()
        featurize_seconds = time.perf_counter() - featurize_start

    total_seconds = time.perf_counter() - total_start
    top_functions = _write_profile_output(profiler, profile_output_path, featurize_seconds)
    return {
        "command": "pair-features",
        "data": dataset_metadata,
        "block": block_key,
        "original_block_size": int(original_block_size),
        "effective_block_size": int(len(papers)),
        "pair_count": int(len(all_pairs)),
        "n_jobs": int(args.n_jobs),
        "chunk_size": int(args.chunk_size),
        "dataset_build_seconds": dataset_metadata["dataset_build_seconds"],
        "featurize_seconds": round(featurize_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "features_shape": [int(features.shape[0]), int(features.shape[1])],
        "labels_shape": [int(labels.shape[0])],
        "peak_rss_gb": round(monitor.peak_gb, 6),
        "rss_scope": monitor.rss_scope,
        "profile_output_path": str(profile_output_path),
        "top_cumulative_functions": top_functions,
        "run_metadata": build_run_metadata(Path(__file__).resolve()),
    }


def run_summary(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = _resolve_path(args.data_root) if args.data_root else _default_data_root()
    if _is_arrow_data_root(data_root):
        manifest = _load_arrow_manifest(data_root)
        block_counts = Counter(_load_arrow_block_counts(data_root))
        largest_block = str(manifest.get("largest_block", {}).get("block") or block_counts.most_common(1)[0][0])
        largest_size = int(manifest.get("largest_block", {}).get("size") or block_counts[largest_block])
        top_block_items = [(largest_block, largest_size)]
        top_block_items.extend(
            (block, size) for block, size in block_counts.most_common(args.top_k) if block != largest_block
        )
        top_block_items = top_block_items[: args.top_k]
        return {
            "command": "summary",
            "data_root": str(data_root),
            "arrow": True,
            "paper_count": int(manifest["counts"]["paper_count"]),
            "largest_block": largest_block,
            "largest_block_size": int(largest_size),
            "largest_block_num_pairs": int(largest_size * (largest_size - 1) // 2),
            "top_blocks": [
                {
                    "block": block,
                    "size": int(size),
                    "num_pairs": int(size * (size - 1) // 2),
                }
                for block, size in top_block_items
            ],
            "run_metadata": build_run_metadata(Path(__file__).resolve()),
        }
    papers = _load_papers_from_root(data_root)
    largest_block, largest_size = _select_largest_block(papers)
    return {
        "command": "summary",
        "data_root": str(data_root),
        "paper_count": int(len(papers)),
        "largest_block": largest_block,
        "largest_block_size": int(largest_size),
        "largest_block_num_pairs": int(largest_size * (largest_size - 1) // 2),
        "top_blocks": _block_summary(papers, top_k=args.top_k),
        "run_metadata": build_run_metadata(Path(__file__).resolve()),
    }


def _emit_result(result: Dict[str, Any], write_json: str) -> None:
    if write_json:
        output_path = _resolve_path(write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote JSON summary: {output_path}")
    print(RESULT_JSON_START)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(RESULT_JSON_END)


def _add_common_data_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default="",
        help="Directory containing papers.json and optional clusters.json. Defaults to data/ when populated, else tests/test_dataset/.",
    )


def _add_prediction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to prod_model.pickle.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes used by S2APLER preprocessing/featurization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000_000,
        help="Clusterer batch_size for pair featurization.",
    )
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=0,
        help="Optional explicit cap on selected block size.",
    )
    parser.add_argument("--profile-output-path", required=True, help="Path for cProfile text output.")
    parser.add_argument("--write-json", default="", help="Optional JSON artifact path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile S2APLER paper-clustering hot paths.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Scan available papers and report largest blocks.")
    _add_common_data_arg(summary)
    summary.add_argument("--top-k", type=int, default=10)
    summary.add_argument("--write-json", default="")

    largest = subparsers.add_parser("largest-block", help="Run prod-model prediction on the largest real block.")
    _add_common_data_arg(largest)
    largest.add_argument("--block", default="", help="Block key. Empty means largest block.")
    largest.add_argument(
        "--block-only",
        action="store_true",
        help="Scan papers.json for the selected block, then preprocess only that block for prediction.",
    )
    largest.add_argument(
        "--arrow-root",
        default="",
        help="Directory containing a S2APLER Arrow bundle. When set, loads only the selected block from Arrow.",
    )
    _add_prediction_args(largest)

    synthetic = subparsers.add_parser(
        "synthetic-large-block",
        help="Replicate a source block to an explicit size and run prod-model prediction.",
    )
    synthetic.add_argument(
        "--source-data-root",
        default="",
        help="Directory containing source papers.json. Defaults to data/ when populated, else tests/test_dataset/.",
    )
    synthetic.add_argument("--block-size", type=int, required=True, help="Explicit synthetic block size.")
    synthetic.add_argument(
        "--source-block",
        default="",
        help="Source block to replicate. Empty means largest source block.",
    )
    _add_prediction_args(synthetic)

    pair_features = subparsers.add_parser(
        "pair-features", help="Profile many_pairs_featurize without model prediction."
    )
    _add_common_data_arg(pair_features)
    pair_features.add_argument("--block", default="", help="Block key. Empty means largest block.")
    pair_features.add_argument("--synthetic-block-size", type=int, default=0)
    pair_features.add_argument("--source-block", default="", help="Source block for --synthetic-block-size.")
    pair_features.add_argument("--max-block-size", type=int, default=0)
    pair_features.add_argument("--max-pairs", type=int, default=0, help="Optional explicit cap on pair count.")
    pair_features.add_argument("--n-jobs", type=int, default=1)
    pair_features.add_argument("--chunk-size", type=int, default=100)
    pair_features.add_argument("--profile-output-path", required=True)
    pair_features.add_argument("--write-json", default="")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        result = run_summary(args)
    elif args.command == "largest-block":
        result = run_largest_block(args)
    elif args.command == "synthetic-large-block":
        result = run_synthetic_large_block(args)
    elif args.command == "pair-features":
        result = run_pair_features(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    _emit_result(result, getattr(args, "write_json", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
