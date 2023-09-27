import abc
from dataclasses import dataclass

from mlflow.pyfunc.spark_model_cache import SparkModelCache
# noinspection PyProtectedMember
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
import numpy as np
import pandas
import pyspark.sql.functions as F
import torch
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import SparkSession, DataFrame

import tidal_algorithmic_mixes.utils.constants as c
from tidal_algorithmic_mixes.etl_model import ETLModel
from tidal_algorithmic_mixes.utils.config import Config
from tidal_algorithmic_mixes.utils.mix_utils import last_n_items


@dataclass
class DiscoveryMixSasRecModelTransformationData:
    inference: DataFrame


@dataclass
class DiscoveryMixSasRecModelTransformationOutput:
    df: DataFrame


class DiscoveryMixSasRecModelTransformationConfig(Config):
    def __init__(self, **kwargs):
        self.max_seq_len = int(kwargs.get("max_seq_len", 500))
        self.batch_size = int(kwargs.get("batch_size", 128))
        self.n_recs = int(kwargs.get("n_recs", 6000))
        self.n_partitions = int(kwargs.get("n_partitions", 60))
        self.inference_device = kwargs.get("inference_device", "cuda:0")  # cpu or cuda:0
        self.model_path = kwargs.get("model_path")
        Config.__init__(self, **kwargs)


class DiscoveryMixSasRecModelTransformation(ETLModel):

    # noinspection PyTypeChecker
    def __init__(self, spark: SparkSession, **kwargs):
        self.spark = spark
        self.conf = DiscoveryMixSasRecModelTransformationConfig(**kwargs)
        self._data: DiscoveryMixSasRecModelTransformationData = None
        self._output: DiscoveryMixSasRecModelTransformationOutput = None
        self.spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", self.conf.batch_size)

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

        user_items = (self.data.inference
                      .withColumn(c.ITEMS, last_n_items(c.ITEMS, F.lit(self.conf.max_seq_len)))
                      .repartition(self.conf.n_partitions))

        predictions = self.predict(user_items,
                                   self.conf.n_recs,
                                   f'runs:/{self.conf.run_id}/script_model',
                                   self.conf.inference_device,
                                   self.conf.model_path)

        self._output = DiscoveryMixSasRecModelTransformationOutput(predictions)

    def predict(self, user_items, n_recs, model_uri, device, path):
        spark_udf = _get_spark_pandas_udf(self.spark, n_recs, model_uri, device, path)
        return user_items.select(F.col(c.USER), spark_udf(c.ITEMS).alias(c.RECOMMENDATIONS))

    @property
    def data(self) -> DiscoveryMixSasRecModelTransformationData:
        return self._data

    @property
    def output(self) -> DiscoveryMixSasRecModelTransformationOutput:
        return self._output


def _get_spark_pandas_udf(spark, n_recs, model_uri, device, path):
    with TempDir() as local_tmpdir:
        local_model_path = _download_artifact_from_uri(
            artifact_uri=model_uri, output_path=local_tmpdir.path()
        )
        archive_path = SparkModelCache.add_local_model(spark, local_model_path)

    vocabulary = spark.read.parquet(f"{path}/vocabulary")

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")  # performance improvement

    ix2item = vocabulary.select(c.ITEM, c.INDEX).toPandas()
    ix2item_bc = spark.sparkContext.broadcast(dict(zip(ix2item[c.INDEX], ix2item[c.ITEM])))

    def predict(items_batch):
        model, _ = SparkModelCache.get_or_load(archive_path)
        max_len = max(
            map(lambda x: len(x), items_batch))  # get the longest sequence length in this batch, used for padding
        items_batch = list(map(lambda x: x.tolist()[-max_len:] + [0] * (max_len - len(x)), items_batch))
        items_batch = torch.tensor(items_batch)
        results = model.predict({c.SEQ_MODEL_ITEMS_SEQ: items_batch,
                                 c.SEQ_MODEL_TOP_K: n_recs,
                                 c.DEVICE: device})
        res = []
        for items in results:
            res.append(np.vectorize(ix2item_bc.value.get)(np.array(items)))
        return pandas.Series(res)

    # noinspection PyTypeChecker
    return pandas_udf(predict, ArrayType(StringType()))
