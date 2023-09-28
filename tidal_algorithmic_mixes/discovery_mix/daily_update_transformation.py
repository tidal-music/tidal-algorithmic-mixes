import abc
import time
import pyspark.sql.functions as F

from dataclasses import dataclass
from datetime import date
from pyspark.sql import SparkSession, DataFrame
from tidal_algorithmic_mixes.etl_model import ETLModel
from tidal_algorithmic_mixes.utils import mix_utils
from tidal_algorithmic_mixes.utils.config import Config
import tidal_algorithmic_mixes.utils.constants as c


@dataclass
class DiscoveryMixDailyUpdateTransformationData:
    mixes: DataFrame


@dataclass
class DiscoveryMixDailyUpdateTransformationOutput:
    df: DataFrame


class DiscoveryMixDailyUpdateTransformationConfig(Config):
    def __init__(self, **kwargs):
        self.current_date = kwargs.get('current_date', date.today())
        self.mix_size = int(kwargs.get('mix_size', 10))
        Config.__init__(self, **kwargs)


class DiscoveryMixDailyUpdateTransformation(ETLModel):
    """
    The main discovery mix pipeline is responsible for outputting a large number of recommendations for each user
    on a weekly basis. This job simply takes a subset of these recommendations each day making sure each user
    gets  10 new recommendations each day. E.g. on Monday we use recommendations from 0-9, Tuesday 10-19, etc.
    """

    # noinspection PyTypeChecker
    def __init__(self, spark: SparkSession, **kwargs):
        self.spark = spark
        self._data: DiscoveryMixDailyUpdateTransformationData = None
        self._output: DiscoveryMixDailyUpdateTransformationOutput = None
        self.config = DiscoveryMixDailyUpdateTransformationConfig(**kwargs)

    @abc.abstractmethod
    def extract(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        discovery_mix = (self.slicer(self.data.mixes,
                                     self.config.current_date,
                                     self.config.mix_size)
                         .withColumn(c.UPDATED, F.lit(mix_utils.updated(time.time())))
                         .where(F.size(c.TRACKS) >= self.config.mix_size - 2))
        self._output = DiscoveryMixDailyUpdateTransformationOutput(discovery_mix)

    def slicer(self, mixes: DataFrame, current_date: date, mix_size: int) -> DataFrame:
        """ Extract the tracks of the day from the weekly computed list """
        offset = self.offset(current_date, mix_size) + 1  # slice starts from 1
        return mixes.withColumn(c.TRACKS, F.slice(c.TRACKS, offset, mix_size))

    @staticmethod
    def offset(current_date, mix_size):
        return current_date.weekday() * mix_size

    @property
    def data(self) -> DiscoveryMixDailyUpdateTransformationData:
        return self._data

    @property
    def output(self) -> DiscoveryMixDailyUpdateTransformationOutput:
        return self._output
