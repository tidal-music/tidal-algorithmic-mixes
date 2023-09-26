import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql.window import Window

import tidal_algorithmic_mixes.utils.constants as c
KNOWN_PERCENTAGE = "knownPercentage"


class DiscoveryMixSortTransformer(Transformer):
    """
    We don't want all the recommended tracks from the artists that are known to a user to be put into a single mix.
    The tracks from known artists should be distributed evenly over the different days of the week
    """
    def __init__(self, precompute_size=70, sort_column=c.POS):
        super(DiscoveryMixSortTransformer, self).__init__()
        self.precompute_size = precompute_size
        self.sort_column = sort_column

    def _transform(self, dataset):
        return (dataset
                .withColumn(KNOWN_PERCENTAGE,
                            F.sum(c.KNOWN_ARTIST).over(Window.partitionBy(c.USER_ID)) / self.precompute_size)
                .withColumn(self.sort_column,
                            F.row_number().over(Window.partitionBy(c.USER_ID, c.KNOWN_ARTIST).orderBy(self.sort_column)))
                .withColumn(self.sort_column,
                            F.when(F.col(c.KNOWN_ARTIST) == 1,
                                   F.col(self.sort_column) * (1 - F.col(KNOWN_PERCENTAGE)))
                            .otherwise(F.col(self.sort_column) * F.col(KNOWN_PERCENTAGE)))
                .drop(KNOWN_PERCENTAGE))
