import logging
import unittest

from pyspark.sql import SparkSession


class PySparkTest(unittest.TestCase):

    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession
                 .builder
                 # fix a bug UnsupportedOperationException
                 .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                 .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                 .appName(__name__)
                 .enableHiveSupport()
                 .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()
