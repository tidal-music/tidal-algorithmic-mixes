# TODO: move to per-transformers

from pyspark.ml.base import Transformer

import tidal_algorithmic_mixes.utils.constants as c


class EnrichWithArtistCluster(Transformer):
    """ Enrich with data from artist clustering model """

    def __init__(self, artist_clusters, artist_id_col=c.ARTIST_ID, artist_clusters_cols=(c.CLUSTER)):
        super().__init__()
        self.artist_clusters = artist_clusters
        self.artist_id_col = artist_id_col
        self.artist_clusters_cols = artist_clusters_cols

    def _transform(self, dataset):
        artist_cluster = (self.artist_clusters
                          .withColumnRenamed(c.ARTIST_ID, self.artist_id_col)
                          .select(self.artist_id_col, self.artist_clusters_cols))

        return dataset.join(artist_cluster, self.artist_id_col, "left_outer")
