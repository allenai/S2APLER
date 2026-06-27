import json

import numpy as np
import pytest

from scripts.convert_to_arrow import main as convert_to_arrow_main
from s2apler.arrow_io import (
    convert_json_dataset_to_arrow,
    load_clusters_from_arrow,
    load_papers_from_arrow,
    validate_arrow_bundle,
)
from s2apler.data import PDData
from s2apler.featurizer import FeaturizationInfo, many_pairs_featurize


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_convert_and_validate_arrow_bundle(tmp_path):
    output_dir = tmp_path / "arrow"

    manifest = convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    validated = validate_arrow_bundle(output_dir, require_clusters=True)
    with open("tests/test_dataset/papers.json", encoding="utf-8") as infile:
        expected_paper_count = len(json.load(infile))
    assert manifest["schema"] == "s2apler_arrow_bundle_v2"
    assert validated["counts"]["paper_count"] == expected_paper_count
    assert validated["counts"]["paper_author_count"] > 0
    assert validated["counts"]["cluster_count"] > 0


def test_convert_rejects_missing_cluster_path_before_writing(tmp_path):
    output_dir = tmp_path / "arrow"

    with pytest.raises(FileNotFoundError, match="Missing clusters JSON"):
        convert_json_dataset_to_arrow(
            papers_path="tests/test_dataset/papers.json",
            clusters_path=tmp_path / "missing_clusters.json",
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_convert_cli_rejects_explicit_missing_cluster_path(tmp_path):
    output_dir = tmp_path / "arrow"

    with pytest.raises(FileNotFoundError, match="Missing clusters JSON"):
        convert_to_arrow_main(
            [
                "--papers",
                "tests/test_dataset/papers.json",
                "--clusters",
                str(tmp_path / "missing_clusters.json"),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert not output_dir.exists()


def test_convert_rejects_empty_papers_json_before_writing(tmp_path):
    papers_path = tmp_path / "papers.json"
    output_dir = tmp_path / "arrow"
    papers_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Papers JSON is empty"):
        convert_json_dataset_to_arrow(papers_path=papers_path, output_dir=output_dir)

    assert not output_dir.exists()


def test_convert_rejects_existing_bundle_files_without_manifest(tmp_path):
    output_dir = tmp_path / "arrow"
    output_dir.mkdir()
    existing_papers_file = output_dir / "papers.arrow"
    existing_papers_file.write_text("do not truncate", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass overwrite=True"):
        convert_json_dataset_to_arrow(
            papers_path="tests/test_dataset/papers.json",
            clusters_path="tests/test_dataset/clusters.json",
            output_dir=output_dir,
        )

    assert existing_papers_file.read_text(encoding="utf-8") == "do not truncate"


def test_arrow_block_load_matches_json_pddata_and_features(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    arrow_papers, metadata = load_papers_from_arrow(output_dir, block_key="reviewerlistfor")
    with open("tests/test_dataset/papers.json", encoding="utf-8") as infile:
        all_papers = json.load(infile)
    json_papers = {paper_id: all_papers[paper_id] for paper_id in arrow_papers}

    json_dataset = PDData(json_papers, name="json", mode="inference", balanced_pair_sample=False)
    arrow_dataset = PDData(arrow_papers, name="arrow", mode="inference", balanced_pair_sample=False)

    assert metadata["selected_block"] == "reviewerlistfor"
    assert list(arrow_dataset.papers) == list(json_dataset.papers)
    assert arrow_dataset.get_blocks() == json_dataset.get_blocks()
    assert arrow_dataset.papers == json_dataset.papers

    papers = list(arrow_dataset.papers)
    pairs = [(papers[i], papers[j], np.nan) for i in range(len(papers)) for j in range(i + 1, len(papers))]
    featurizer = FeaturizationInfo()
    json_features, _, _ = many_pairs_featurize(pairs, json_dataset, featurizer, 1, False, 100, nan_value=-1)
    arrow_features, _, _ = many_pairs_featurize(pairs, arrow_dataset, featurizer, 1, False, 100, nan_value=-1)
    np.testing.assert_allclose(arrow_features, json_features)


def test_pddata_from_arrow_loads_all_papers(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    json_dataset = PDData(
        "tests/test_dataset/papers.json",
        name="json",
        mode="inference",
        balanced_pair_sample=False,
    )
    raw_arrow_papers, raw_metadata = load_papers_from_arrow(output_dir)
    arrow_dataset, metadata = PDData.from_arrow(
        str(output_dir),
        name="arrow",
        mode="inference",
        balanced_pair_sample=False,
    )

    assert raw_metadata["load_all"]
    assert set(raw_arrow_papers) == set(json_dataset.papers)
    assert metadata["load_all"]
    assert arrow_dataset.papers == json_dataset.papers


def test_arrow_full_load_rejects_block_cap_without_selection(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="max_block_size requires block_key"):
        load_papers_from_arrow(output_dir, max_block_size=3)

    with pytest.raises(ValueError, match="max_block_size requires block_key"):
        load_papers_from_arrow(output_dir, paper_ids=["1"], max_block_size=3)


def test_arrow_load_rejects_paper_ids_with_block_key(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_papers_from_arrow(output_dir, paper_ids=["1"], block_key="reviewerlistfor")


def test_pddata_constructor_uses_arrow_bundle_when_path_passed(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    json_dataset = PDData(
        "tests/test_dataset/papers.json",
        clusters="tests/test_dataset/clusters.json",
        name="json",
        balanced_pair_sample=False,
    )
    arrow_dataset = PDData(
        str(output_dir),
        clusters=str(output_dir),
        name="arrow",
        balanced_pair_sample=False,
    )

    assert arrow_dataset.arrow_metadata["load_all"]
    assert len(arrow_dataset.papers) == len(json_dataset.papers)
    assert arrow_dataset.paper_to_cluster_id == json_dataset.paper_to_cluster_id
    assert arrow_dataset.papers == json_dataset.papers


def test_pddata_constructor_does_not_auto_load_arrow_clusters(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    json_dataset = PDData("tests/test_dataset/papers.json", name="json", balanced_pair_sample=False)
    arrow_dataset = PDData(str(output_dir), name="arrow", balanced_pair_sample=False)

    assert arrow_dataset.papers == json_dataset.papers
    assert arrow_dataset.clusters is None
    assert arrow_dataset.paper_to_cluster_id is None


def test_explicit_arrow_clusters_require_cluster_file(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="missing file key 'clusters'"):
        PDData(
            str(output_dir),
            clusters=str(output_dir),
            name="arrow",
            balanced_pair_sample=False,
        )

    with pytest.raises(ValueError, match="missing file key 'clusters'"):
        PDData.from_arrow(str(output_dir), name="arrow", clusters=True)


def test_arrow_load_respects_max_block_size_sorting(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    arrow_papers, metadata = load_papers_from_arrow(output_dir, block_key="reviewerlistfor", max_block_size=3)
    with open("tests/test_dataset/papers.json", encoding="utf-8") as infile:
        all_papers = json.load(infile)
    expected_ids = sorted(
        paper_id for paper_id, paper in all_papers.items() if paper.get("block") == "reviewerlistfor"
    )[:3]

    assert metadata["original_block_size"] == 5
    assert list(arrow_papers) == expected_ids


def test_arrow_clusters_round_trip(tmp_path):
    output_dir = tmp_path / "arrow"
    convert_json_dataset_to_arrow(
        papers_path="tests/test_dataset/papers.json",
        clusters_path="tests/test_dataset/clusters.json",
        output_dir=output_dir,
    )

    clusters = load_clusters_from_arrow(output_dir)

    assert clusters is not None
    assert clusters["PM_79887"]["sourced_paper_ids"] == [20797514, 210097756]


def test_arrow_preserves_json_author_order(tmp_path):
    papers = {
        "p1": {
            "title": "Example One",
            "authors": [
                {"first": "Second", "last": "Author", "middle": [], "position": 1},
                {"first": "First", "last": "Author", "middle": [], "position": 0},
            ],
            "block": "b",
        }
    }
    papers_path = tmp_path / "papers.json"
    output_dir = tmp_path / "arrow"
    _write_json(papers_path, papers)
    convert_json_dataset_to_arrow(papers_path=papers_path, output_dir=output_dir)

    arrow_papers, _metadata = load_papers_from_arrow(output_dir)
    json_dataset = PDData(papers, name="json", mode="inference", balanced_pair_sample=False)
    arrow_dataset = PDData(arrow_papers, name="arrow", mode="inference", balanced_pair_sample=False)

    assert [author.author_info_first for author in arrow_dataset.papers["p1"].authors] == ["Second", "First"]
    assert arrow_dataset.papers == json_dataset.papers


def test_arrow_preserves_json_cluster_keys_empty_clusters_and_id_types(tmp_path):
    papers = {
        "001": {"title": "One", "authors": [], "block": "b"},
        "2": {"title": "Two", "authors": [], "block": "b"},
    }
    clusters = {
        "outer_key": {
            "cluster_id": "inner_id",
            "sourced_paper_ids": ["001"],
            "model_version": -1,
        },
        "numeric_membership": {
            "cluster_id": "numeric_membership",
            "sourced_paper_ids": [2],
            "model_version": -1,
        },
        "empty_cluster": {
            "cluster_id": "empty_inner",
            "sourced_paper_ids": [],
            "model_version": -1,
        },
    }
    papers_path = tmp_path / "papers.json"
    clusters_path = tmp_path / "clusters.json"
    output_dir = tmp_path / "arrow"
    _write_json(papers_path, papers)
    _write_json(clusters_path, clusters)

    manifest = convert_json_dataset_to_arrow(
        papers_path=papers_path,
        clusters_path=clusters_path,
        output_dir=output_dir,
    )
    arrow_clusters = load_clusters_from_arrow(output_dir)
    json_dataset = PDData(papers, clusters=clusters, name="json", balanced_pair_sample=False)
    arrow_dataset = PDData(
        str(output_dir),
        clusters=str(output_dir),
        name="arrow",
        balanced_pair_sample=False,
    )

    assert manifest["counts"]["cluster_count"] == 3
    assert manifest["counts"]["cluster_membership_count"] == 2
    assert arrow_clusters == clusters
    assert arrow_dataset.clusters == json_dataset.clusters
    assert arrow_dataset.paper_to_cluster_id == json_dataset.paper_to_cluster_id
