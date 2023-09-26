# TODO: move to per-transformers

from pyspark.ml.base import Transformer
from pyspark.pandas import DataFrame

import tidal_algorithmic_mixes.utils.constants as c


class ArtistMetaDataEnricherTransformer(Transformer):

    def __init__(self, artist_metadata: DataFrame, artist_id_col=c.ARTIST_ID, cols=(c.NAME, c.IMAGE)):
        self.artist_metadata = artist_metadata
        self.artist_id_col = artist_id_col
        self.cols = list(cols)
        super().__init__()

    def _transform(self, dataset):
        index = (self.artist_metadata
                 .withColumnRenamed(c.ID, self.artist_id_col)
                 .withColumnRenamed(c.PICTURE, c.IMAGE)
                 .select([self.artist_id_col] + self.cols))

        return dataset.join(index, self.artist_id_col)
