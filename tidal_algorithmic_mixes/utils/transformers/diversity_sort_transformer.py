# TODO: move to per-transformers

from pyspark.ml.base import Transformer
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import tidal_algorithmic_mixes.utils.constants as c


class DiversitySortTransformer(Transformer):
    """
    Sorts a mix in an 'intelligent' manner by e.g. spacing out tracks from the same artists or whatever you
    pass using the partition arguments.

    :param id_col:          Your id, this could e.g. be a mixId or userId or another partition we want to sort for
    :param partition_one:   The first partition you would like space out (e.g. artistId)
    :param partition_two:   The second partition you would like to space out (e.g. trackBundleId)
    :param order_by:        Column containing the original relevance based ordering
    :param gap:             The number of spaces to add between each artist/album recommendation
    :return:                A DataFrame with a new 'rank' column that can be used to sort the list
    """

    def __init__(self, id_col, partition_one, partition_two, order_by, gap=5):
        super(DiversitySortTransformer, self).__init__()
        self.id_col = id_col
        self.partition_one = self.id_col + partition_one
        self.partition_two = self.id_col + partition_two
        self.order_by = order_by
        self.gap = gap

    def _transform(self, dataset):
        w1 = Window.partitionBy(self.partition_one).orderBy(self.order_by)
        w1_rank_window = Window.partitionBy(self.id_col).orderBy("w1_first_rank")
        w2 = Window.partitionBy(self.partition_two).orderBy(self.order_by)
        w2_rank_window = Window.partitionBy(self.id_col).orderBy("w2_first_rank")

        ordered = (dataset
                   .withColumn("w1_first_rank", F.first(self.order_by, True).over(w1) - 1)
                   .withColumn("w2_first_rank", F.first(self.order_by, True).over(w2) - 1)
                   .withColumn("w1_rank", F.dense_rank().over(w1_rank_window) - 1)
                   .withColumn("w2_rank", F.dense_rank().over(w2_rank_window) - 1)
                   .withColumn("w1_inter_rank", F.row_number().over(w1) - 1)
                   .withColumn("w2_inter_rank", F.row_number().over(w2) - 1)
                   .withColumn("ordering", (F.least(F.col("w1_rank"), F.col("w2_rank"))
                                            + self.gap * F.greatest(F.col("w1_inter_rank"), F.col("w2_inter_rank"))))
                   .withColumn(c.RANK, F.row_number().over(Window.partitionBy(self.id_col).orderBy(
                    F.col('ordering'), F.desc(self.order_by))))
                   .drop("w1_inter_rank", "w2_inter_rank", "w2_rank", "w1_rank", "w1_first_rank", "w2_first_rank",
                         "ordering"))
        return ordered
