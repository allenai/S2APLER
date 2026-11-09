"""Correction notices stay apart from the papers they correct (allenai/scholar#42255).

The end-to-end cases are real production requests where notices and articles share titles,
authors and venues; Crossref's update-to field confirms each notice corrects its article.
"""

import json
import os

import numpy as np
import pytest

from s2apler.data import PDData
from s2apler.model import _condensed_pair_chunks
from s2apler.text import is_correction_notice, marks_notice_identifiers, normalize_text

FIXTURE = "tests/test_dataset/correction_notices.json"


@pytest.mark.parametrize(
    ("title", "notice", "marks"),
    [
        ("Correction to: Optimal Disclosure on Crowdfunding Platforms", True, True),
        ("Correction: Optimal Disclosure on Crowdfunding Platforms", True, True),
        ("Corrigendum: A Hydrogel Drink With High Fructose Content", True, True),
        ("Erratum: Behçet syndrome: a contemporary view", True, True),
        ("Publisher Correction: Is wave-particle objectivity compatible", True, True),
        ("Author Correction: Something measured", True, True),
        ("Addendum: More data", True, True),
        ("Erratum", True, False),
        ("Correction", True, False),
        ("Retraction Note: Upregulation of the long non-coding RNA", True, False),
        ("Expression of Concern: Epothilone B benefits", True, False),
        ("RETRACTED: Some article", False, False),
        ("Notice of Retraction: Some article", False, False),
        ("Correction of hallux valgus", False, False),
        ("Correctional facilities and health", False, False),
        ("Behçet syndrome: a contemporary view", False, False),
    ],
)
def test_notice_titles(title, notice, marks):
    normalized = normalize_text(title)
    assert is_correction_notice(normalized) is notice
    assert marks_notice_identifiers(normalized) is marks


def dataset_of(records):
    papers = {
        key: {"authors": [{"first": "Ann", "last": "Lee"}], "block": "b", "year": 2020, **record}
        for key, record in records.items()
    }
    return PDData(papers, name="notices", mode="inference")


def test_copies_of_a_named_notice_count_as_notices():
    dataset = dataset_of(
        {
            "notice": {"title": "Correction: That gut feeling", "doi": "10.1/notice"},
            "notice_copy": {"title": "That gut feeling", "doi": "10.1/notice"},
            "notice_by_pmid": {"title": "That gut feeling", "pmid": "7", "doi": None},
            "notice_pmid": {"title": "Correction: That gut feeling", "pmid": "7", "doi": "10.1/notice"},
            "article": {"title": "That gut feeling", "doi": "10.1/article"},
            "article_pdf": {"title": "That gut feeling"},
        }
    )
    assert dataset.correction_notices == {"notice", "notice_copy", "notice_by_pmid", "notice_pmid"}
    assert dataset.get_constraint("notice_copy", "article") == 1e4
    assert dataset.get_constraint("article", "article_pdf") is None


@pytest.mark.parametrize("title", ["Correction", "Retraction note: That gut feeling", "Expression of concern: X"])
def test_notice_titles_sharing_the_article_doi_count_as_the_article(title):
    # Such records sometimes carry the corrected or retracted article's own DOI. The DOI forces
    # them into the article's cluster, so marking them would push the article's copies out.
    dataset = dataset_of(
        {
            "notice": {"title": title, "doi": "10.1/article"},
            "article": {"title": "That gut feeling", "doi": "10.1/article"},
            "article_pdf": {"title": "That gut feeling"},
        }
    )
    assert dataset.correction_notices == set()
    assert dataset.get_constraint("notice", "article") == 0
    assert dataset.get_constraint("notice", "article_pdf") is None


def test_groups_holding_only_notice_titles_are_notices():
    dataset = dataset_of(
        {
            "retraction": {"title": "Retraction note: That gut feeling", "doi": "10.1/notice", "pdf_hash": "n"},
            "retraction_pdf": {"title": None, "pdf_hash": "n"},
            "bare": {"title": "Erratum"},
            "article": {"title": "That gut feeling", "doi": "10.1/article"},
        }
    )
    assert dataset.correction_notices == {"retraction", "retraction_pdf", "bare"}


def test_array_constraints_match_per_pair_rules_with_notices():
    with open(FIXTURE, encoding="utf-8") as f:
        cases = json.load(f)
    for records in cases.values():
        dataset = PDData(records, name="notices", mode="inference")
        papers = sorted(dataset.papers)
        for left, right, first in _condensed_pair_chunks(len(papers), 50):
            values = dataset.constraint_values(papers, left, right, first)
            for k, (i, j) in enumerate(zip(left, right)):
                expected = dataset.get_constraint(papers[i], papers[j])
                assert (np.isnan(values[k]) and expected is None) or values[k] == expected


@pytest.fixture(scope="module")
def predictor():
    if not os.path.exists("data/prod_model.pickle"):
        pytest.skip("needs the tracked production model")
    from s2apler.timo.interface import Predictor, PredictorConfig

    return Predictor(PredictorConfig(n_jobs=1), "data")


@pytest.mark.parametrize(
    ("case", "article_doi", "notice_doi"),
    [
        ("crowdfunding", "10.1007/s11151-026-10048-2", "10.1007/s11151-026-10061-5"),
        ("hydrogel", "10.3389/fnut.2020.00088", "10.3389/fnut.2020.00128"),
        ("behcet", "10.1038/nrrheum.2017.208", "10.1038/nrrheum.2018.3"),
        ("gut_feeling", "10.1038/nrc.2016.122", "10.1038/nrc.2016.147"),
    ],
)
def test_real_notices_cluster_apart_from_their_articles(predictor, case, article_doi, notice_doi):
    from s2apler.timo.interface import Instance

    with open(FIXTURE, encoding="utf-8") as f:
        records = json.load(f)[case]
    prediction = predictor.predict_one(Instance(papers=records)).prediction
    cluster_of = {pid: key for key, members in prediction.items() for pid in members}
    assert sorted(cluster_of) == sorted(records)
    article = {cluster_of[pid] for pid, r in records.items() if (r["doi"] or "").lower() == article_doi}
    notice = {cluster_of[pid] for pid, r in records.items() if (r["doi"] or "").lower() == notice_doi}
    # Every copy of each work stays together, and the two works stay apart.
    assert len(article) == 1 and len(notice) == 1 and article != notice


def test_a_notice_carrying_the_article_doi_keeps_the_article_whole(predictor):
    # A MAG "Retraction Note" record carries the retracted article's DOI. The article's copies
    # without identifiers must stay in the article's cluster, apart from the real notice.
    from s2apler.timo.interface import Instance

    with open(FIXTURE, encoding="utf-8") as f:
        records = json.load(f)["mast_cells"]
    prediction = predictor.predict_one(Instance(papers=records)).prediction
    cluster_of = {pid: key for key, members in prediction.items() for pid in members}
    assert sorted(cluster_of) == sorted(records)
    article = {"2038844007", "3310699750", "67028303", "1524897677", "1824594924"}
    copies = {"3286700761", "1459784667", "1431315993", "1444693022"}
    notice = {"74276376", "1824706971", "1996170033", "2039593257"}
    assert len({cluster_of[pid] for pid in article | copies}) == 1
    assert len({cluster_of[pid] for pid in notice}) == 1
    assert cluster_of["2038844007"] != cluster_of["74276376"]


def test_notice_status_follows_shared_pdf_hashes():
    # The erratum's PDF carries the article's title but shares the erratum record's PDF hash.
    dataset = dataset_of(
        {
            "erratum": {"title": "Erratum: Behcet syndrome", "doi": "10.1/erratum"},
            "erratum_publisher": {"title": "Behcet syndrome", "doi": "10.1/erratum", "pdf_hash": "e"},
            "erratum_pdf": {"title": "Behcet syndrome", "pdf_hash": "e"},
            "article": {"title": "Behcet syndrome", "doi": "10.1/article", "pdf_hash": "a"},
            "article_pdf": {"title": "Behcet syndrome", "pdf_hash": "a"},
        }
    )
    assert dataset.correction_notices == {"erratum", "erratum_publisher", "erratum_pdf"}
