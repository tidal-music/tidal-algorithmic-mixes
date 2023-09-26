import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import Window
from tidal_per_transformers.transformers import TopItemsTransformer

import tidal_algorithmic_mixes.utils.constants as c


NR_KNOWN_ARTIST = "nr_known_artist"
NR_UNKNOWN_ARTIST = "nr_unknown_artist"


class SplitByKnownArtistsTransformer(Transformer):
    """ Keep only a % of known artists

    :param mix_size: number of tracks to keep
    :param threshold_known_artists: how much of known artist

    """

    def __init__(self, mix_size, threshold_known_artists):
        super(SplitByKnownArtistsTransformer, self).__init__()
        self.mix_size = mix_size
        self.threshold_known_artists = threshold_known_artists

    def _transform(self, dataset):
        window = Window.partitionBy(c.USER_ID)
        dataset = dataset.withColumn(NR_KNOWN_ARTIST, F.sum(c.KNOWN_ARTIST).over(window))
        dataset = dataset.withColumn("total_artists", F.count(c.ARTIST_ID).over(window))
        dataset = dataset.withColumn(NR_UNKNOWN_ARTIST, F.expr(f"total_artists - {NR_KNOWN_ARTIST}")).drop("total_artists")

        optimal_max_known_artists = int(self.threshold_known_artists * self.mix_size)
        optimal_max_unknown_artists = int(self.mix_size - optimal_max_known_artists)

        max_known_artists = F.expr(
            f"if({optimal_max_unknown_artists}>{NR_UNKNOWN_ARTIST}, {self.mix_size}-{NR_UNKNOWN_ARTIST}, {self.mix_size}-{optimal_max_unknown_artists} )")

        # for users without enough known artists, need more unknown artists
        max_unknown_artists = F.expr(
            f"if({optimal_max_known_artists}>{NR_KNOWN_ARTIST}, {self.mix_size}-{NR_KNOWN_ARTIST}, {self.mix_size}-{optimal_max_known_artists} )")

        known_artist_recs = (TopItemsTransformer(c.USER_ID, F.col(c.POS), max_known_artists)
                             .transform(dataset.where(f"{c.KNOWN_ARTIST}=1")))

        unknown_artist_recs = (TopItemsTransformer(c.USER_ID, F.col(c.POS), max_unknown_artists)
                               .transform(dataset.where(f"{c.KNOWN_ARTIST}=0")))

        return known_artist_recs.unionAll(unknown_artist_recs).drop(NR_KNOWN_ARTIST)
