"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""


import logging
import sys
import unittest
import json
from os.path import join
from s2apler.consts import PROJECT_ROOT_PATH
from .interface import Instance, Prediction


try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning(
        """
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """
    )
    sys.exit(0)

# load data
with open(join(PROJECT_ROOT_PATH, "s2apler", "timo", "papers.json")) as f:
    PAPERS = json.load(f)

with open(join(PROJECT_ROOT_PATH, "s2apler", "timo", "clusters.json")) as f:
    CLUSTERS = json.load(f)

with open(join(PROJECT_ROOT_PATH, "s2apler", "timo", "cluster_seeds.json")) as f:
    CLUSTER_SEEDS = json.load(f)


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        instances = [Instance(papers=PAPERS, cluster_seeds=CLUSTER_SEEDS)]
        predictions = container.predict_batch(instances)[0]
        preds_sub = [v for v in predictions if v["cluster_id"] == "nondestructivetablethardnesstesting_2"][0]

        self.assertEqual(set(preds_sub["paper_ids"]), {"1591643905", "2459452638", "2468186458"})


# instance = Instance(papers=PAPERS, cluster_seeds=CLUSTER_SEEDS)
# instances = [instance]
# predictor = Predictor(config=PredictorConfig(n_jobs=4, predict_incremental=False), artifacts_dir="artifacts")
# predictions = predictor.predict_batch(instances)[0]
# predictor = Predictor(config=PredictorConfig(n_jobs=4, predict_incremental=True), artifacts_dir="artifacts")
# preds_incremental = predictor.predict_batch(instances)
