# TODO: move to per-transformers

from pyspark.ml.base import Transformer
from pyspark.sql import functions as F, DataFrame
from tidal_per_transformers.transformers.utils.spark_utils import get_top_items

import tidal_algorithmic_mixes.utils.constants as c


class MainArtistCompoundMappingTransformer(Transformer):
    """
    Map the artist compound id's to a single main artist (e.g. Miguel feat. Travis Scott -> Miguel)

    :returns    DataFrame where the compound id's have been mapped to their constituent parts
    """
    def __init__(self, artist_compound_mapping: DataFrame):
        super().__init__()
        self.artist_compound_mapping = artist_compound_mapping

    def _transform(self, dataset):
        compound_map = (self.artist_compound_mapping
                        .where("mainartist = 'true'")
                        .withColumnRenamed(c.ARTIST_COMPOUND_ID, c.RESOLVED_ARTIST_ID)
                        .drop(c.ID))

        # Unfortunately the compound table contain duplicates (multiple main artists), keep only 1 (lowest priority)
        deduped = get_top_items(compound_map, [c.ARTIST_ID], c.PRIORITY, 1)

        # If there is no compound entry we already have the main artist
        joined = dataset.join(deduped, c.ARTIST_ID, "left")

        mapped = (joined
            .withColumn(c.ARTIST_ID, F.when(
                F.col(c.RESOLVED_ARTIST_ID).isNull(), F.col(c.ARTIST_ID))
                    .otherwise(F.col(c.RESOLVED_ARTIST_ID)))
            .drop(c.RESOLVED_ARTIST_ID, c.PRIORITY, c.MAIN_ARTIST, c.ARTIST_COMPOUND_ID))

        return mapped
