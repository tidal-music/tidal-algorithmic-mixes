from datetime import date

import tidal_algorithmic_mixes.utils.constants as c

from test.pyspark_test import PySparkTest
from tidal_algorithmic_mixes.discovery_mix.daily_update_transformation import DiscoveryMixDailyUpdateTransformation, \
    DiscoveryMixDailyUpdateTransformationData


class DiscoveryMixDailyUpdateTransformationTestInterface(DiscoveryMixDailyUpdateTransformation):
    def extract(self, *args, **kwargs):
        ...

    def validate(self, *args, **kwargs):
        ...

    def load(self, *args, **kwargs):
        ...


class DiscoveryMixDailyUpdateTest(PySparkTest):

    def test_slicer(self):
        mixes = self.spark.createDataFrame([
            (0, [10, 11, 12, 13, 14, 15, 16]),
            (1, [10, 11, 12, 13, 14, 15, 16]),
            (2, [10, 11, 12, 13, 14, 15, 16]),
            (3, [10, 11, 12, 13, 14, 15, 16])
        ], [c.USER, c.TRACKS])

        runner = DiscoveryMixDailyUpdateTransformationTestInterface(self.spark)

        runner._data = DiscoveryMixDailyUpdateTransformationData(mixes)

        self.assertEqual(runner.slicer(mixes, date(2021, 2, 15), 1).collect()[0][c.TRACKS][0], 10)
        self.assertEqual(runner.slicer(mixes, date(2021, 2, 18), 1).collect()[0][c.TRACKS][0], 13)
        self.assertEqual(runner.slicer(mixes, date(2021, 2, 21), 1).collect()[0][c.TRACKS][0], 16)

    def test_offset(self):
        runner = DiscoveryMixDailyUpdateTransformationTestInterface(self.spark)
        self.assertEqual(runner.offset(date(2021, 2, 15), 10), 0)
        self.assertEqual(runner.offset(date(2021, 2, 18), 10), 30)
        self.assertEqual(runner.offset(date(2021, 2, 21), 10), 60)
