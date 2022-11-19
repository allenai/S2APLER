# S2APLER:  Semantic Scholar (S2) Agglomeration of Papers with Low Error Rate
This repository provides access to the paper clustering dataset and production model. This work was done by Shaurya Rohatgi (dataset mostly) and Sergey Feldman (model mostly), with lots of help from many folks, including: Doug Downey, Regan Huff, Rodney Kinney and Caroline Wu.

The model will be live on semanticscholar.org.

## TODO
- Does `predict_incremental` work as we expect?
- Should the constraints be turned off by default?
- Do we need `altered_cluster_signatures`?
- How do we test the constraints that eng will make?
- Publish model with `tt`: https://github.com/allenai/timo/blob/main/docs/timo-tools/userguide.md

## Installation
To install this package, run the following:

```bash
git clone https://github.com/allenai/S2APLER.git
cd S2APLER
conda create -y --name s2apler python==3.8.15
conda activate s2apler
pip install -r requirements.in
pip install -e .
```

If you run into cryptic errors about GCC on macOS while installing the requirments, try this instead:
```bash
CFLAGS='-stdlib=libc++' pip install -r requirements.in
```

## Data 
To obtain the paper clustering dataset, run the following command after the package is installed (from inside the `S2APLER` directory):  
```[Expected download size is: 3.1 GiB]```

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/paper_clustering data/`

Note that this software package comes with tools specifically designed to access and model the dataset.

## Configuration
Modify the config file at `data/path_config.json`. This file should look like this
```
{
    "main_data_dir": "absolute path to wherever you downloaded the data to",
}
```
As the dummy file says, `main_data_dir` should be set to the location of wherever you downloaded the data to.
You can leave it blank and it will default to the data directory of the package itself.

## How to use S2APLER for loading data and training a model
Once you have downloaded the datasets, you can go ahead and load it up:

```python
from os.path import join
from s2apler.data import PDData
from s2apler.consts import CONFIG

dataset = PDData(
    join(CONFIG["main_data_dir"], "papers.json"),
    clusters=join(CONFIG["main_data_dir"], "clusters.json"),
    name="paper_clustering_dataset",
    n_jobs=8,
    balanced_pair_sample=False,
    train_pairs_size=5000000,
    val_pairs_size=5000000,
    test_pairs_size=5000000,
)
```

This will take ~20m - there is a lot of text pre-processing to do.

The first step in the S2APLER pipeline is to specify a featurizer and then train a binary classifier
that tries to guess whether two signatures are referring to the same person. 

We'll do hyperparameter selection with the validation set and then get the test area under ROC curve.

Here's how to do all that:

```python
from s2apler.model import PairwiseModeler
from s2apler.featurizer import FeaturizationInfo, featurize
from s2apler.eval import pairwise_eval

featurization_info = FeaturizationInfo()
# the cache will make it faster to train multiple times - it stores the features on disk for you
train, val, test = featurize(dataset, featurization_info, n_jobs=8, use_cache=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

# here is the pairwise model!
pairwise_model = PairwiseModeler(n_iter=25, n_jobs=8)
# this does hyperparameter selection, which is why we need to pass in the validation set.
pairwise_model.fit(X_train, y_train, X_val, y_val)

# this will also dump a lot of useful plots (ROC, PR, SHAP) to the figs_path
pairwise_metrics = pairwise_eval(X_test, y_test, pairwise_model.classifier, figs_path='figs/', title='example')
print(pairwise_metrics)
```

The second stage in the S2APLER pipeline is to tune hyperparameters for the clusterer on the validation data
and then evaluate the full clustering pipeline on the test blocks.

We use agglomerative clustering as implemented in `fastcluster` with average linkage.
There is only one hyperparameter to tune.

```python
from s2apler.model import Clusterer, FastCluster
from hyperopt import hp

clusterer = Clusterer(
    featurization_info,
    pairwise_model,
    cluster_model=FastCluster(linkage="average"),
    search_space={"eps": hp.uniform("eps", 0, 1)},
    n_iter=25,
    n_jobs=8,
)
clusterer.fit(dataset)

# the metrics_per_signature are there so we can break out the facets if needed
metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
print(metrics)
```

That's pretty much it! Please see the script that fits the production model and saves it to disk: `scripts/dump_pairwise_model.py`. This script also serves as a tutorial.

## How to use S2APLER for predicting with a saved model
Assuming you have a clusterer already fit, you can dump the model to disk like so
```python
import pickle

with open("saved_model.pkl", "wb") as _pkl_file:
    pickle.dump(clusterer, _pkl_file)
```

You can then reload it, load a new dataset, and run prediction
```python
import pickle

with open("saved_model.pkl", "rb") as _pkl_file:
    clusterer = pickle.load(_pkl_file)

# load some PDData here
pddata = PDData()  # etc.
pred_clusters, pred_distance_matrices = clusterer.predict(anddata.get_blocks(), pddata)
```

Our S2 production models are in the `data` folder.

### Incremental prediction
There is a also a `predict_incremental` function on the `Clusterer`, that allows prediction for just a small set of *new* signatures. When instantiating `ANDData`, you can pass in `cluster_seeds`, which will be used instead of model predictions for those signatures. If you call `predict_incremental`, the full distance matrix will not be created, and the new signatures will simply be assigned to the cluster they have the lowest average distance to, as long as it is below the model's `eps`, or separately reclustered with the other unassigned signatures, if not within `eps` of any existing cluster.

## Licensing
The code in this repo is released under the Apache 2.0 license (license included in the repo. The dataset is released under ODC-BY (included in S3 bucket with the data). We would also like to acknowledge that some of the affiliations data comes directly from the Microsoft Academic Graph (https://aka.ms/msracad).

## AI2
S2APLER is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

Thanks to Mike D'Arcy for the name.
