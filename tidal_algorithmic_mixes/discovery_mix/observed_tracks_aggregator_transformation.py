
import abc
from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

import tidal_algorithmic_mixes.utils.constants as c
from tidal_algorithmic_mixes.etl_model import ETLModel


@dataclass
class ObservedDiscoveryMixTracksAggregatorTransformationData:
    observed_mixes: DataFrame
    mixes: DataFrame


@dataclass
class ObservedDiscoveryMixTracksAggregatorTransformationOutput:
    df: DataFrame


class ObservedDiscoveryMixTracksAggregatorTransformation(ETLModel):
    """
    Daily storing of tracks that a user has observed after opening a discovery mix
    """

    # noinspection PyUnusedLocal,PyTypeChecker
    def __init__(self, spark: SparkSession, **kwargs):
        self.spark = spark
        self._data: ObservedDiscoveryMixTracksAggregatorTransformationData = None
        self._output: ObservedDiscoveryMixTracksAggregatorTransformationOutput = None

    @abc.abstractmethod
    def extract(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        pass

    def transform(self):
        """
        Fetch tracks for the mixes that have been observed
        """
        self._output = ObservedDiscoveryMixTracksAggregatorTransformationOutput(
            self.data.mixes
            .join(self.data.observed_mixes, c.MIX_ID)
            .select(c.USER, F.explode(c.TRACKS).alias(c.TRACK_GROUP)))

    @property
    def data(self) -> ObservedDiscoveryMixTracksAggregatorTransformationData:
        return self._data

    @property
    def output(self) -> ObservedDiscoveryMixTracksAggregatorTransformationOutput:
        return self._output
