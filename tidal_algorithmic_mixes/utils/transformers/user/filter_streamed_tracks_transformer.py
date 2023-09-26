# TODO: move to per-transformers

from datetime import datetime

import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta
from pyspark.ml import Transformer

import tidal_algorithmic_mixes.utils.constants as c


class FilterStreamedTracksTransformer(Transformer):

    def __init__(self,
                 track_dataset,
                 playback_threshold: int = 1,
                 recency_threshold: int = 12,
                 join_columns=[c.USER_ID, c.TRACK_GROUP],
                 last_stream_date_column: str = c.LAST_STREAMED_DATE,
                 filter_streamed_tracks=True):
        """
        :param track_dataset:             dataset containing the user track listening history
        :param playback_threshold:        hard limit on # streams before a track is considered known
        :param recency_threshold:         for lower stream counts we can apply a recency filter
        :param join_columns:              columns used to remove known tracks
        :param last_stream_date_column:   column for track last stream date
        :param filter_streamed_tracks:    toggle sorting on/off for pipelines where this is configurable
        """
        super(FilterStreamedTracksTransformer, self).__init__()
        self.filter_streamed_tracks = filter_streamed_tracks
        self.track_dataset = track_dataset
        self.join_columns = join_columns
        self.playback_threshold = playback_threshold
        self.recency_threshold = recency_threshold
        self.last_stream_date_column = last_stream_date_column

    def _transform(self, dataset):
        if self.filter_streamed_tracks and self.track_dataset is not None:
            if self.playback_threshold and self.recency_threshold:
                memory_window = datetime.now() - relativedelta(months=self.recency_threshold)
                self.track_dataset = (self.track_dataset
                                      .where((F.col(c.COUNT) >= self.playback_threshold) |
                                             (F.col(self.last_stream_date_column) >= memory_window)))

            return dataset.join(self.track_dataset, on=self.join_columns, how="left_anti")
        else:
            return dataset
