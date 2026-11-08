import os

import pytest

from s2apler.native_scorer import NativeLightGBMClassifier
from s2apler.timo.interface import Predictor, PredictorConfig

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/prod_model.pickle"), reason="needs the tracked production model"
)


@pytest.mark.parametrize("n_jobs, threads", [(1, 1), (3, 3), (-1, os.cpu_count() or 1)])
def test_scores_natively_with_resolved_n_jobs(n_jobs, threads):
    predictor = Predictor(PredictorConfig(n_jobs=n_jobs), "data")
    assert isinstance(predictor.clusterer.classifier, NativeLightGBMClassifier)
    assert predictor.clusterer.classifier.num_threads == threads


def test_rejects_zero_n_jobs_at_load():
    with pytest.raises(ValueError, match="n_jobs"):
        Predictor(PredictorConfig(n_jobs=0), "data")
