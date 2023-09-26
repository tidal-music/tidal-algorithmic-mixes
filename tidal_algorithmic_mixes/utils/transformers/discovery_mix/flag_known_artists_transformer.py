from datetime import datetime

import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta
from pyspark.ml import Transformer

import tidal_algorithmic_mixes.utils.constants as c


class FlagKnownArtistsTransformer(Transformer):
    """ Flag the dataset with artists that are known/previously observed by a user """
    def __init__(self,
                 all_time_tracks,
                 artist_playback_threshold: int,
                 recency_threshold: int,
                 fav_artists=None,
                 last_stream_date_column: str = c.LAST_STREAMED_DATE):
        """
        :param all_time_tracks:             all time history of user
        :param artist_playback_threshold:   hard limit for # streams before an artist is considered as known/observed
        :param last_stream_date_column:     column for track last stream date
        :param fav_artists:                 df containing user favourite artists
        :param recency_threshold:           months since last stream before we "forget" lower stream counts
        """
        super().__init__()
        self.all_time_tracks = all_time_tracks
        self.playback_threshold = artist_playback_threshold
        self.recency_threshold = datetime.now() - relativedelta(months=recency_threshold)
        self.fav_artists = fav_artists
        self.last_stream_date_column = last_stream_date_column

    def _transform(self, dataset):
        known_artists = (self.all_time_tracks
                         .where((F.col(c.COUNT) >= self.playback_threshold)
                                | (F.col(self.last_stream_date_column) >= self.recency_threshold))
                         .select(c.USER_ID, c.ARTIST_ID, F.lit(1).alias(c.KNOWN_ARTIST)))

        if self.fav_artists is not None:
            fav_artists = self.fav_artists.select(c.USER_ID, c.ARTIST_ID, F.lit(1).alias(c.KNOWN_ARTIST))
            known_artists = known_artists.union(fav_artists).distinct()

        return (dataset.join(known_artists, [c.USER_ID, c.ARTIST_ID], "left_outer")
                .select(c.USER_ID, c.ARTIST_ID, c.TRACK_GROUP, c.KNOWN_ARTIST, c.POS)
                .na.fill(0, c.KNOWN_ARTIST))
