import abc
from datetime import datetime

import pyspark.sql.functions as F
from tidal_per_transformers.transformers import WithColumnRenamedTransformer, JoinTransformer, CleanTextTransformer, \
    TrackGroupAvailabilityByCountryTransformer, TopItemsTransformer, AggregateTransformer, SelectTransformer
from tidal_per_transformers.transformers.single_dag_pipeline import SingleDAGPipeline as Pipeline

import tidal_algorithmic_mixes.utils.constants as c

from dataclasses import dataclass
from pyspark.sql import SparkSession, DataFrame
from tidal_algorithmic_mixes.etl_model import ETLModel
from tidal_algorithmic_mixes.utils.config import Config
from tidal_algorithmic_mixes.utils.transformers.artist.enrich_with_artist_cluster import EnrichWithArtistCluster
from tidal_algorithmic_mixes.utils.transformers.blacklist.user_blacklist_filter_transformer import \
    UserBlacklistFilterTransformer
from tidal_algorithmic_mixes.utils.transformers.discovery_mix.discovery_mix_output_transformer import \
    DiscoveryMixOutputTransformer
from tidal_algorithmic_mixes.utils.transformers.discovery_mix.discovery_mix_sort_transformer import \
    DiscoveryMixSortTransformer
from tidal_algorithmic_mixes.utils.transformers.discovery_mix.flag_known_artists_transformer import \
    FlagKnownArtistsTransformer
from tidal_algorithmic_mixes.utils.transformers.discovery_mix.split_by_known_artists_transformer import \
    SplitByKnownArtistsTransformer
from tidal_algorithmic_mixes.utils.transformers.diversity_sort_transformer import DiversitySortTransformer
from tidal_algorithmic_mixes.utils.transformers.posexplode_transformer import PosExplodeTransformer
from tidal_algorithmic_mixes.utils.transformers.user.filter_streamed_tracks_transformer import \
    FilterStreamedTracksTransformer


@dataclass
class DiscoveryMixPostProcessorTransformationData:
    tracks_metadata: DataFrame
    track_groups_metadata: DataFrame
    precomputed_recs: DataFrame
    user_history_tracks: DataFrame
    user_history_artists: DataFrame
    user_fav_tracks: DataFrame
    user_fav_artists: DataFrame
    artist_clusters: DataFrame
    user_observed_tracks: DataFrame
    user_table: DataFrame
    user_blacklist_table: DataFrame
    artist_compound_mapping_table: DataFrame


@dataclass
class DiscoveryMixPostProcessorTransformationOutput:
    df: DataFrame


class DiscoveryMixPostProcessorTransformationConfig(Config):

    def __init__(self, **kwargs):
        self.mix_size = int(kwargs.get('mix_size', 70))
        self.min_mix_size = int(kwargs.get('min_mix_size', 30))
        self.threshold_known_artists = float(kwargs.get('threshold_known_artists', 0.2))
        self.max_artist_items = int(kwargs.get('max_artist_items', 1))
        self.known_artist_streams = int(kwargs.get('known_artist_streams', 3))
        self.known_artist_recency = int(kwargs.get('known_artist_recency', 6))
        self.known_track_streams = int(kwargs.get('known_track_streams', 2))
        self.known_track_recency = int(kwargs.get('known_track_recency', 12))
        self.now = datetime.utcnow().date().isoformat()

        Config.__init__(self, **kwargs)


class DiscoveryMixPostProcessorTransformation(ETLModel):

    # noinspection PyTypeChecker
    def __init__(self, spark: SparkSession, **kwargs):
        self.spark = spark
        self.conf = DiscoveryMixPostProcessorTransformationConfig(**kwargs)
        self._data: DiscoveryMixPostProcessorTransformationData = None
        self._output: DiscoveryMixPostProcessorTransformationOutput = None

    @abc.abstractmethod
    def extract(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        pass

    def transform(self):
        user_country = DiscoveryMixPostProcessorTransformation.get_user_country(self.data.user_table)
        track_group_metadata = DiscoveryMixPostProcessorTransformation.get_track_group_metadata(
            self.data.tracks_metadata)
        track_group_available_countries = DiscoveryMixPostProcessorTransformation.get_track_group_available_countries(
            self.data.track_groups_metadata)

        pipeline = Pipeline(stages=[
            WithColumnRenamedTransformer(c.USER, c.USER_ID),
            PosExplodeTransformer(explode_col=c.RECOMMENDATIONS, alias=c.TRACK_GROUP),
            JoinTransformer(user_country, on=c.USER_ID),
            JoinTransformer(track_group_metadata, on=c.TRACK_GROUP),
            CleanTextTransformer(output_col=c.CLEANED_TITLE),
            UserBlacklistFilterTransformer(self.data.user_blacklist_table,
                                           self.data.artist_compound_mapping_table,
                                           filter_track=True,
                                           filter_artist=True,
                                           user_col=c.USER_ID),
            TrackGroupAvailabilityByCountryTransformer(track_group_available_countries),
            FilterStreamedTracksTransformer(self.data.user_observed_tracks,
                                            playback_threshold=0,
                                            recency_threshold=0,
                                            last_stream_date_column=c.DT),
            FilterStreamedTracksTransformer(self.data.user_history_tracks,
                                            playback_threshold=self.conf.known_track_streams,
                                            recency_threshold=self.conf.known_track_recency,
                                            last_stream_date_column=c.DT),
            FilterStreamedTracksTransformer(self.data.user_history_tracks,
                                            playback_threshold=self.conf.known_track_streams,
                                            recency_threshold=self.conf.known_track_recency,
                                            join_columns=[c.USER_ID, c.CLEANED_TITLE],
                                            last_stream_date_column=c.DT),
            FilterStreamedTracksTransformer(self.data.user_fav_tracks,
                                            playback_threshold=0,
                                            recency_threshold=0,
                                            last_stream_date_column=c.DT),
            FlagKnownArtistsTransformer(self.data.user_history_artists,
                                        self.conf.known_artist_streams,
                                        self.conf.known_artist_recency,
                                        self.data.user_fav_artists,
                                        last_stream_date_column=c.DT),
            TopItemsTransformer([c.USER_ID, c.ARTIST_ID],
                                F.col(c.POS),
                                self.conf.max_artist_items),
            EnrichWithArtistCluster(self.data.artist_clusters),
            SplitByKnownArtistsTransformer(self.conf.mix_size,
                                           self.conf.threshold_known_artists),
            DiversitySortTransformer([c.USER_ID],
                                     [c.CLUSTER],
                                     [c.ARTIST_ID],
                                     c.POS,
                                     gap=int(self.conf.mix_size / 7)),
            # spread out the clusters
            DiscoveryMixSortTransformer(self.conf.mix_size, sort_column=c.RANK),
            DiscoveryMixOutputTransformer(self.conf.now, sort_column=c.RANK, min_mix_size=self.conf.min_mix_size)
        ])
        output = pipeline.fit(self.data.precomputed_recs).persist()
        self._output = DiscoveryMixPostProcessorTransformationOutput(output)

    @staticmethod
    def get_user_country(user_table: DataFrame):
        """Returns user id and user country

        :param user_table: user table df
        :return: user id and user country DF
        """
        user_country_pipeline = Pipeline(stages=[
            SelectTransformer([c.ID, c.COUNTRY_CODE]),
            WithColumnRenamedTransformer(c.ID, c.USER_ID)])
        return user_country_pipeline.fit(user_table)

    @staticmethod
    def get_track_group_metadata(tracks_metadata: DataFrame):
        """Returns track group and all main artists

        :param tracks_metadata: tracks metadata fs table
        :return: track group and all main artists
        """
        track_group_artist_pipeline = Pipeline(stages=[
            AggregateTransformer(c.TRACK_GROUP,
                                 [F.first(c.MAIN_ARTISTS_IDS).getItem(0).alias(c.ARTIST_ID),
                                  F.first(c.TITLE).alias(c.TITLE)]),
        ])
        return track_group_artist_pipeline.fit(tracks_metadata)

    @staticmethod
    def get_track_group_available_countries(track_group_metadata: DataFrame):
        """Gets all available countries for a track group

        :param track_group_metadata: tracks group metadata fs table
        :return: track group available countries
        """
        select = SelectTransformer([c.TRACK_GROUP, c.AVAILABLE_COUNTRY_CODES])
        return select.transform(track_group_metadata)

    @property
    def data(self) -> DiscoveryMixPostProcessorTransformationData:
        return self._data

    @property
    def output(self) -> DiscoveryMixPostProcessorTransformationOutput:
        return self._output
