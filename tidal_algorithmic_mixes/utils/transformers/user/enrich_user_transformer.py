# TODO: move to per-transformers

from pyspark.ml import Transformer
from pyspark.sql import DataFrame


class EnrichUserTransformer(Transformer):
    """ Join the dataset with fields from the user table.
    """
    def __init__(self, user_table: DataFrame, userid_col: str, mapping=None):
        """
        :param userid_col: Name of the dataset column containing the user id.
        :param mapping: List of columns to include in the join from the user table. Optionally a dict can be passed
          with mappings from user table columns to new column names that will be joined with the dataset.
        :type  mapping: tuple[str]|List[str]|dict[str, str]
        """
        super().__init__()
        self.user_table = user_table
        self.userid_col = userid_col

        if isinstance(mapping, dict):
            self.mapping = mapping
        elif isinstance(mapping, (tuple, list)):
            self.mapping = {k: k for k in mapping}
        else:
            raise ValueError()

    def _transform(self, dataset):
        user_table = self.user_table.withColumnRenamed('id', self.userid_col)

        for k, v in self.mapping.items():
            user_table = user_table.withColumnRenamed(k, v)

        return dataset.join(user_table, self.userid_col, 'left')
