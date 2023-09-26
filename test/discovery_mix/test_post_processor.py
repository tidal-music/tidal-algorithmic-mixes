from datetime import datetime
from pyspark.sql.types import Row
from pyspark_test import PySparkTest
from tidal_algorithmic_mixes.discovery_mix.post_processor_transformation import\
    DiscoveryMixPostProcessorTransformation, DiscoveryMixPostProcessorTransformationData


class DiscoveryMixPostProcessorTransformationTestInterface(DiscoveryMixPostProcessorTransformation):
    def extract(self, *args, **kwargs):
        ...

    def validate(self, *args, **kwargs):
        ...

    def load(self, *args, **kwargs):
        ...


class DiscoveryMixPostProcessorTest(PySparkTest):

    def setUp(self):
        tracks_metadata = self.spark.createDataFrame([Row(id=1,
                                                          title='Chime again',
                                                          popularityWW=0,
                                                          trackNumber=16,
                                                          volumeNumber=1,
                                                          numAlbums=3,
                                                          explicit=False,
                                                          generatedFromVideo=False,
                                                          trackGroup='xxx',
                                                          audioQuality='LOSSLESS',
                                                          available=True,
                                                          version='x',
                                                          duration=192,
                                                          mixes={'x': 'y'},
                                                          mainArtistsIds=[1],
                                                          mainArtistsNames=['Me'],
                                                          mainArtistId=1,
                                                          mainArtistPicture='xxx',
                                                          featuringArtistsIds=[''],
                                                          albumId=1,
                                                          masterBundleId='x',
                                                          albumTitle='Victorian',
                                                          albumCover='be7c307bc938',
                                                          releaseDate=datetime(2010, 6, 8, 0, 0),
                                                          albumReleaseDate=datetime(2010, 6, 8, 0, 0),
                                                          creditsArtistId=[1],
                                                          creditsName=['La La'],
                                                          creditsRole=['Main Artist'],
                                                          creditsRoleCategory=['HIDDEN'],
                                                          numTrackStreams=0,
                                                          numTrackStreamers=0,
                                                          voicenessScore=0,
                                                          voice=1,
                                                          genre='Christmas',
                                                          originalGenre='Christmas',
                                                          AvailableCountryCodes=['AD', 'AE'])])
        track_groups_metadata = self.spark.createDataFrame([Row(trackGroup='xxx',
                                                                AvailableCountryCodes=['AD', 'AE'])])

        precomputed_recs = self.spark.createDataFrame([Row(user=1,
                                                           recommendations=['xxx', 'xyz'])])

        user_history_tracks = self.spark.createDataFrame([Row(userId=1,
                                                              productId=2,
                                                              artistId=2,
                                                              trackGroup='xyz',
                                                              title="Don't Let The Sun Go Down On Me",
                                                              cleanedTitle="dd",
                                                              count=2,
                                                              source='UserTracksHistory',
                                                              dt=datetime(2020, 12, 21, 13, 3, 36, 534000))])
        user_history_artists = self.spark.createDataFrame([Row(userId=1,
                                                               artistId=3,
                                                               count=10,
                                                               source='UserArtistsHistory',
                                                               dt=datetime(2022, 5, 2, 20, 28, 23, 516000))])
        user_fav_tracks = self.spark.createDataFrame([Row(userId=1,
                                                          productId=5,
                                                          artistId=7,
                                                          trackGroup='aaa',
                                                          title='Breathing Underwater',
                                                          cleanedTitle='aa',
                                                          count=1,
                                                          source='UserTracksFavourite',
                                                          dt=datetime(2020, 10, 23, 6, 49, 33))])
        user_fav_artists = self.spark.createDataFrame([Row(userId=1,
                                                           artistId=111,
                                                           count=1,
                                                           source='UserArtistsFavourite',
                                                           dt=datetime(2019, 11, 21, 13, 31, 11))])

        artist_clusters = self.spark.createDataFrame([Row(artistId=1, cluster=42)])

        user_observed_tracks = self.spark.createDataFrame([Row(userId=1,
                                                               productId=5,
                                                               artistId=7,
                                                               trackGroup='aaa',
                                                               title='Breathing Underwater',
                                                               cleanedTitle='aa',
                                                               count=1,
                                                               source='UserTracksDiscoveryObserved',
                                                               dt=datetime(2020, 10, 23, 6, 49, 33))])

        user_table = self.spark.createDataFrame([Row(id=1, countrycode='AD')])

        user_blacklist_table = self.spark.createDataFrame([Row(artifactId='111',
                                                               artifactType='TRACK',
                                                               created=1568546619349,
                                                               userId='3')])

        artist_compound_mapping_table = self.spark.createDataFrame([Row(id=4,
                                                                        artistid=5,
                                                                        artistcompoundid=6,
                                                                        priority=1,
                                                                        mainartist=False)])

        self.data = DiscoveryMixPostProcessorTransformationData(tracks_metadata,
                                                                track_groups_metadata,
                                                                precomputed_recs,
                                                                user_history_tracks,
                                                                user_history_artists,
                                                                user_fav_tracks,
                                                                user_fav_artists,
                                                                artist_clusters,
                                                                user_observed_tracks,
                                                                user_table,
                                                                user_blacklist_table,
                                                                artist_compound_mapping_table
                                                                )

    def test_transform(self):
        post_processor = DiscoveryMixPostProcessorTransformationTestInterface(self.spark,
                                                                              threshold_known_artists=1,
                                                                              mix_size=1,
                                                                              min_mix_size=0)
        post_processor._data = self.data
        post_processor.transform()
        res = post_processor.output.output.collect()[0]
        self.assertEqual(Row(user=1, tracks=['xxx'], mixId='1f1451b3b417516e9e4b4423958', atDate=res.atDate),
                         res)
