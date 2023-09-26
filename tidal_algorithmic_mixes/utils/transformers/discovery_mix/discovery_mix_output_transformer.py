import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import Window
from pyspark.sql.types import IntegerType, StringType

import tidal_algorithmic_mixes.utils.constants as c
from tidal_algorithmic_mixes.utils import mix_utils


class DiscoveryMixOutputTransformer(Transformer):

    def __init__(self, today, sort_column=c.POS, min_mix_size=66):
        super(DiscoveryMixOutputTransformer, self).__init__()
        self.today = today
        self.sort_column = sort_column
        self.min_mix_size = min_mix_size

    def _transform(self, dataset):
        w = Window.partitionBy(c.USER_ID).orderBy(self.sort_column)

        return (dataset
                .withColumn(c.TRACK_GROUP, F.col(c.TRACK_GROUP).astype(StringType()))
                .withColumn(c.TRACKS, F.collect_list(c.TRACK_GROUP).over(w))
                .groupBy(c.USER_ID)
                .agg(F.max(c.TRACKS).alias(c.TRACKS))
                .where(F.size(c.TRACKS) >= self.min_mix_size)
                .withColumnRenamed(c.USER_ID, c.USER)
                .withColumn(c.USER, F.col(c.USER).astype(IntegerType()))
                .withColumn(c.MIX_ID, mix_utils.mix_id(F.lit("discovery_mix_"), F.col(c.USER)))
                .withColumn(c.AT_DATE, F.lit(self.today).cast(StringType())))
