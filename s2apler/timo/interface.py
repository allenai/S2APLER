"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

import pickle
from os.path import join
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, BaseSettings, Field
from s2apler.data import PDData


class Author(BaseModel):
    author_info_first: Optional[str] = Field(required=False)
    author_info_middle: Optional[str] = Field(required=False)
    author_info_last: Optional[str] = Field(required=False)
    author_info_suffix: Optional[str] = Field(required=False)


class Paper(BaseModel):
    title: Optional[str] = Field(required=False)
    abstract: Optional[str] = Field(required=False)
    authors: Optional[List[Author]] = Field(default_factory=list)
    venue: Optional[str] = Field(required=False)
    journal_name: Optional[str] = Field(required=False)
    year: Optional[int] = Field(required=False)
    sourced_paper_id: str = Field(required=True, description="paper id from the paper sources table")
    corpus_paper_id: Optional[str] = Field(required=False, description="paper id *after* clustering")
    doi: Optional[str] = Field(required=False)
    pmid: Optional[str] = Field(required=False)
    source_id: Optional[str] = Field(required=False)
    pdf_hash: Optional[str] = Field(required=False)
    source: Optional[str] = Field(required=False)
    block: Optional[str] = Field(required=False)


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    papers: Dict[str, Paper]
    cluster_seeds: Optional[Union[str, Dict]] = Field(
        required=False,
        description="Pairs of source paper ids to merge or keep separate. "
                    "Defining this means clustering in incremental mode."
    )


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    prediction: Dict[str, List[str]]


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.

    The fields below are examples only; please replace them with ones
    appropriate for your model. These serve as a record of the ENV
    vars the consuming application needs to set.
    """

    n_jobs: int = Field(default=4, description="number of jobs to use for parallelization", required=False)
    use_default_constraints_as_supervision: bool = Field(
        default=True,
        description="Whether to use the default constraints when constructing the distance matrices. These are high precision and can save a lot of compute/time.",
        required=False,
    )
    dont_merge_cluster_seeds: bool = Field(
        default=True,
        description="Controls whether to use cluster seeds to enforce 'dont merge' as well as 'must merge' constraints",
        required=False,
    )
    eps: float = Field(
        default=None,
        description="epsilon for the clusterer. If None, will use what comes with the model",
        required=False,
    )


class Predictor:
    """
    Interface on to your underlying model.

    This class is instantiated at application startup as a singleton.
    You should initialize your model inside of it, and implement
    prediction methods.

    If you specified an artifacts.tar.gz for your model, it will
    have been extracted to `artifacts_dir`, provided as a constructor
    arg below.
    """

    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        with open(join(self._artifacts_dir, "prod_model.pickle"), "rb") as f:
            self.clusterer = pickle.load(f)["clusterer"]
        if self._config.eps is not None:
            self.clusterer.set_params({"eps": self._config.eps})
        self.clusterer.use_default_constraints_as_supervision = self._config.use_default_constraints_as_supervision
        self.clusterer.dont_merge_cluster_seeds = self._config.dont_merge_cluster_seeds

    def predict_one(self, instance: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        dataset = PDData(
            instance.dict()["papers"],
            cluster_seeds=instance.dict()["cluster_seeds"],
            name="test_dataset",
            mode="inference",
            n_jobs=self._config.n_jobs,
        )
        if instance.cluster_seeds:
            out = self.clusterer.predict_incremental(dataset.papers, dataset)
        else:
            out = self.clusterer.predict(dataset.get_blocks(), dataset)[0]
        return Prediction(prediction=out)

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.

        If your model gets performance benefits from batching during inference,
        implement that here, explicitly.

        Otherwise, you can leave this method as-is and just implement
        `predict_one()` above. The default implementation here passes
        each Instance into `predict_one()`, one at a time.

        The size of the batches passed into this method is configurable
        via environment variable by the calling application.
        """
        return [self.predict_one(instance) for instance in instances]
