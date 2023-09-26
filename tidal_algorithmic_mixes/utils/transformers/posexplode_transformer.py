# TODO: move to per-transformers

import pyspark.sql.functions as F
from pyspark.ml.base import Transformer


class PosExplodeTransformer(Transformer):
    """ Returns a new row for each element with position in the given array or map
    :type alias: str|dict
    """
    def __init__(self, explode_col=None, alias='col'):
        super(PosExplodeTransformer, self).__init__()
        self.explode_col = explode_col
        self.alias = alias

    def _transform(self, dataset):
        columns = [col for col in dataset.columns if self.explode_col not in col]
        exploded = dataset.select(F.posexplode(self.explode_col), *columns)

        if type(self.alias) is dict:
            for k, v in self.alias.items():
                exploded = exploded.withColumn(v, F.col(k))
            exploded = exploded.drop('col')
        else:
            exploded = exploded.withColumnRenamed('col', self.alias)

        return exploded
