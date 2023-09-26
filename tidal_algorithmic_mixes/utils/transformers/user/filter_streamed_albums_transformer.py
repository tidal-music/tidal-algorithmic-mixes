# TODO: move to per-transformers

import pyspark.sql.functions as F
from pyspark.ml import Transformer

import tidal_algorithmic_mixes.utils.constants as c


class FilterStreamedAlbumsTransformer(Transformer):

    """
    Filter album that have been listened more than the threshold filter_min_album_streams
    """

    def __init__(self, filter_min_album_streams, user_history):
        super().__init__()
        self.filter_min_album_streams = filter_min_album_streams
        self.user_history = user_history

    def _transform(self, dataset):
        user_history = (self.user_history
                        .groupBy(c.USER_ID, c.MASTER_BUNDLE_ID)
                        .agg(F.sum("count").alias(c.COUNT))
                        .where(f"count>={self.filter_min_album_streams}")
                        .drop("count"))

        return dataset.join(user_history, on=[c.USER_ID, c.MASTER_BUNDLE_ID], how="left_anti")
