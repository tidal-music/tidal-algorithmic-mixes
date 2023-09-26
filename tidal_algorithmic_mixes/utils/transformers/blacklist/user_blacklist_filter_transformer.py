# TODO: move to per-transformers

import pyspark.sql.functions as F
from pyspark.ml.base import Transformer
from pyspark.sql import DataFrame
import tidal_algorithmic_mixes.utils.constants as c


ARTIFACT_ID = "artifactId"
TRACK = "TRACK"
VIDEO = "VIDEO"
ARTIST = "ARTIST"


class UserBlacklistFilterTransformer(Transformer):
    """ Transformer for removing content blacklisted by users from a DataFrame.
    """
    def __init__(self,
                 user_blacklist: DataFrame,  # backup_tables.get_user_blacklist_table(self.sc)
                 compound_table: DataFrame,  # backup_tables.get_artist_compound_mapping_table(self.sc)
                 filter_track: bool = True,
                 filter_artist: bool = True,
                 filter_video: bool = False,
                 user_col: str = c.USER_ID):
        super().__init__()
        self.filter_track = filter_track
        self.filter_video = filter_video
        self.filter_artist = filter_artist
        self.user_col = user_col
        self.user_blacklist = user_blacklist
        self.compound_table = compound_table

    def _transform(self, dataset):

        if self.user_col != c.USER_ID:
            self.user_blacklist = self.user_blacklist.withColumnRenamed(c.USER_ID, self.user_col)

        if self.filter_track:
            dataset = self.filter_blacklisted_tracks(dataset, self.user_blacklist)

        if self.filter_video:
            dataset = self.filter_blacklisted_videos(dataset, self.user_blacklist)

        if self.filter_artist:
            dataset = self.filter_blacklisted_artists(dataset, self.user_blacklist)

        return dataset

    def filter_blacklisted_tracks(self, dataset, blacklist):
        """
        Remove blacklisted tracks from the dataset. Maps the productIds -> trackGroups before removing the
        blacklisted content from the dataset.

        :type dataset:      pyspark.sql.DataFrame
        :param blacklist:   DataFrame containing all blacklisted content
        :type blacklist:    pyspark.sql.DataFrame
        """
        track_blacklist = (blacklist
                            .where(F.col(c.ARTIFACT_TYPE) == TRACK)
                            .select(self.user_col, F.col(ARTIFACT_ID).alias(c.TRACK_GROUP)))

        return dataset.join(track_blacklist, [self.user_col, c.TRACK_GROUP], "left_anti")

    def filter_blacklisted_videos(self, dataset, blacklist):
        """
        Remove blacklisted videos from the dataset.

        :type dataset:      pyspark.sql.DataFrame
        :param blacklist:   DataFrame containing all blacklisted content
        :type blacklist:    pyspark.sql.DataFrame
        """
        video_blacklist = (blacklist
            .where(F.col(c.ARTIFACT_TYPE) == VIDEO)
            .withColumnRenamed(ARTIFACT_ID, c.VIDEO_ID))

        return dataset.join(video_blacklist, [self.user_col, c.VIDEO_ID], "left_anti")

    def filter_blacklisted_artists(self, dataset, blacklist):
        """
        Remove blacklisted artists from the dataset. Filters any tracks by the artist or any compound the artist
        is part of.

        :type dataset:      pyspark.sql.DataFrame
        :param blacklist:   DataFrame containing all blacklisted content
        :type blacklist:    pyspark.sql.DataFrame
        """
        main_artist_blacklist = (blacklist
            .where(F.col(c.ARTIFACT_TYPE) == ARTIST)
            .select(F.col(self.user_col), F.col(ARTIFACT_ID).alias(c.ARTIST_ID))
            .persist())

        compound_table = (self.compound_table
                          .withColumnRenamed(c.ARTIST_ID, c.COMPOUND_ID))

        compound_blacklist = (main_artist_blacklist
            .join(compound_table, main_artist_blacklist[c.ARTIST_ID] == compound_table[c.ARTIST_COMPOUND_ID])
            .select(F.col(self.user_col), F.col(c.COMPOUND_ID).alias(c.ARTIST_ID)))

        # Keep both the main artists and any compounds they are part of
        full_blacklist = main_artist_blacklist.union(compound_blacklist)

        return dataset.join(full_blacklist, [self.user_col, c.ARTIST_ID], "left_anti")
