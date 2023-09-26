# TODO: move to per-transformers

from pyspark.ml.base import Transformer
from pyspark.sql import functions as F

import tidal_algorithmic_mixes.utils.constants as c


class EnrichWithSimilarArtists(Transformer):
    """
    Enrich with artists from the artist clustering model
    it will add a column ENRICHED=1 and  ENRICHED_POSITION=neighbour position
    """

    def __init__(self, similar_artists, max_similars):
        super().__init__()
        self.similar_artists = similar_artists
        self.max_similars = max_similars

    def _transform(self, dataset):
        similar_artists_exploded = (self.similar_artists
                                    .withColumn(c.NEIGHBOURS, F.slice(F.col(c.NEIGHBOURS), 1, self.max_similars))
                                    .select(c.ARTIST_ID, F.posexplode(c.NEIGHBOURS)))

        return (dataset.join(similar_artists_exploded, c.ARTIST_ID)
                .withColumnRenamed("pos", c.ENRICHED_POSITION)  # pos from posexplode
                .withColumn(c.ARTIST_ID, F.col("col"))  # col from posexplode
                .drop(c.NEIGHBOURS, "col")
                .withColumn(c.ENRICHED, F.lit(1)))
