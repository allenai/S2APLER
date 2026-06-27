from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from s2apler.arrow_io import convert_json_dataset_to_arrow


def _load_profile_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "s2apler_profile.py"
    spec = importlib.util.spec_from_file_location("s2apler_profile_for_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_largest_block_counts_papers():
    profile = _load_profile_module()
    block, size = profile._select_largest_block(
        {
            "1": {"block": "a"},
            "2": {"block": "b"},
            "3": {"block": "b"},
        }
    )
    assert block == "b"
    assert size == 2


def test_select_block_from_papers_uses_largest_or_explicit_block():
    profile = _load_profile_module()
    papers = {
        "1": {"block": "a"},
        "2": {"block": "b"},
        "3": {"block": "b"},
        "4": {"block": "b"},
    }

    block, paper_ids, original_size = profile._select_block_from_papers(papers, "", 2)
    assert block == "b"
    assert paper_ids == ["2", "3"]
    assert original_size == 3

    block, paper_ids, original_size = profile._select_block_from_papers(papers, "a", 0)
    assert block == "a"
    assert paper_ids == ["1"]
    assert original_size == 1


def test_make_synthetic_papers_requires_explicit_large_block_size():
    profile = _load_profile_module()
    try:
        profile._make_synthetic_papers({"1": {"block": "a", "authors": []}}, 1)
    except ValueError as exc:
        assert "greater than 1" in str(exc)
    else:
        raise AssertionError("expected block-size validation")


def test_make_synthetic_papers_clears_hard_merge_ids():
    profile = _load_profile_module()
    synthetic = profile._make_synthetic_papers(
        {
            "1": {
                "block": "a",
                "authors": [],
                "doi": "10.1/x",
                "pmid": "pmid",
                "pdf_hash": "hash",
                "source_id": "source-id",
                "source_uris": ["https://example.test/a.pdf"],
            }
        },
        3,
        source_block="a",
    )
    assert list(synthetic) == [
        "synthetic_0000000",
        "synthetic_0000001",
        "synthetic_0000002",
    ]
    assert {paper["block"] for paper in synthetic.values()} == {profile.SYNTHETIC_BLOCK_KEY}
    assert all(paper["doi"] is None for paper in synthetic.values())
    assert all(paper["pmid"] is None for paper in synthetic.values())
    assert all(paper["pdf_hash"] is None for paper in synthetic.values())
    assert all(paper["source_uris"] is None for paper in synthetic.values())


def test_make_capped_pairs_stops_at_max_pairs():
    profile = _load_profile_module()

    class GuardedPapers:
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            if index > 4:
                raise AssertionError("pair generation ignored --max-pairs")
            return str(index)

    pairs = profile._make_capped_pairs(GuardedPapers(), max_pairs=3)

    assert [(left, right) for left, right, _label in pairs] == [
        ("0", "1"),
        ("0", "2"),
        ("0", "3"),
    ]
    assert all(np.isnan(label) for _left, _right, label in pairs)


def test_make_capped_pairs_rejects_negative_cap():
    profile = _load_profile_module()

    try:
        profile._make_capped_pairs(["1", "2"], max_pairs=-1)
    except ValueError as exc:
        assert "cannot be negative" in str(exc)
    else:
        raise AssertionError("expected negative cap validation")


def test_extract_marked_json_payload():
    profile = _load_profile_module()
    payload = {"command": "summary", "paper_count": 2}
    stdout = f"noise\n{profile.RESULT_JSON_START}\n{json.dumps(payload)}\n{profile.RESULT_JSON_END}\n"
    assert profile.extract_marked_json_payload(stdout) == payload


def test_summary_uses_arrow_when_data_root_is_arrow_bundle(tmp_path):
    profile = _load_profile_module()
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    result = profile.run_summary(argparse.Namespace(data_root=str(output_dir), top_k=3))

    assert result["arrow"]
    assert result["paper_count"] == 170
    assert result["largest_block"] == "preface"
