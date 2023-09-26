import tidal_algorithmic_mixes.utils.constants as c
from pyspark_test import PySparkTest
from tidal_algorithmic_mixes.discovery_mix.observed_tracks_aggregator_transformation import \
    ObservedDiscoveryMixTracksAggregatorTransformation, ObservedDiscoveryMixTracksAggregatorTransformationData


class ObservedDiscoveryMixTracksAggregatorTransformationTestInterface(
        ObservedDiscoveryMixTracksAggregatorTransformation):
    def extract(self, *args, **kwargs):
        ...

    def validate(self, *args, **kwargs):
        ...

    def load(self, *args, **kwargs):
        ...


class ObservedDiscoveryMixTracksAggregatorTest(PySparkTest):

    def setUp(self):
        super().setUp()

    def test_transform(self):
        user_1 = 26129743
        user_2 = 43727840

        user_1_mix = "5b5b0f74b66cbecf46de5f00297"
        user_2_mix = "c71e7c0b5f8daeaff1bdea48f9f"

        tracks_user_1 = [1, 2, 3, 4, 5, 6]
        tracks_user_2 = [3, 4, 5, 6, 7, 9]

        mixes = self.spark.createDataFrame([
            (user_1_mix,),
            (user_2_mix,),
        ], [c.MIX_ID])

        observed_mixes = self.spark.createDataFrame([
            (user_1_mix, user_1, tracks_user_1),
            (user_2_mix, user_2, tracks_user_2),
            ("xvxfewfwsdf34r3sf3jfaae4tgs", 1664, [11, 22, 33, 44]),
            ("a71e7xffw4rzdzdf34zsz23ead3", 1984, [55, 66, 77, 11, 22]),
        ], [c.MIX_ID, c.USER, c.TRACKS])

        runner = ObservedDiscoveryMixTracksAggregatorTransformationTestInterface(self.spark)
        runner._data = ObservedDiscoveryMixTracksAggregatorTransformationData(observed_mixes=observed_mixes,
                                                                              mixes=mixes)
        runner.transform()
        res = runner.output.output

        self.assertEqual(res.columns,  [c.USER, c.TRACK_GROUP])
        self.assertEqual(res.count(), len(tracks_user_1) + len(tracks_user_2))

        self.assertEqual([user_1, user_2], ([x[c.USER] for x in res.select(c.USER).distinct().collect()]))
