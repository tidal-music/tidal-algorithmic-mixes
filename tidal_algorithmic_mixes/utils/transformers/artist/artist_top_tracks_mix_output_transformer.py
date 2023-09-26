# TODO: move to per-transformers

import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import Window
from pyspark.sql.types import StringType

import tidal_algorithmic_mixes.utils.constants as c
from tidal_algorithmic_mixes.utils import mix_utils


class ArtistTopTracksMixOutputTransformer(Transformer):

    def __init__(self, today, sort_column=c.RANK):
        super(ArtistTopTracksMixOutputTransformer, self).__init__()
        self.today = today
        self.sort_column = sort_column

    def _transform(self, dataset):
        dataset = dataset.withColumn(c.TRACK_GROUP, F.col(c.TRACK_GROUP).astype(StringType()))

        w = Window.partitionBy(c.ARTIST_ID).orderBy(self.sort_column)

        return (dataset
                .withColumn(c.TRACKS, F.collect_list(c.TRACK_GROUP).over(w))
                .groupBy(c.ARTIST_ID)
                .agg(F.max(c.TRACKS).alias(c.TRACKS))
                .withColumnRenamed(c.USER_ID, c.USER)
                .withColumn(c.ARTIST_ID, F.col(c.ARTIST_ID).astype('int'))
                .withColumn(c.MIX_ID, mix_utils.mix_id(F.lit(self.get_prefix()), F.col(c.ARTIST_ID)))
                .withColumn(c.AT_DATE, F.lit(self.today).cast(StringType())))
                # .withColumn(c.AT_DATE, F.lit("latest"))) # TODO change to latest when doing PER-1686

    def get_prefix(self):
        return "artist_top_tracks_mix_"
