import abc
import mlflow
import tidal_algorithmic_mixes.utils.constants as c

from typing import Dict


class ETLModel(abc.ABC):

    @abc.abstractmethod
    def extract(self, *args, **kwargs):
        """ Read the data from the source tables.
        """

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        """ Apply the transformations to the dataset. Use a pipeline if possible.
        """

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        """ Load the data to the end target
        """

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        """ Validate data, generally using great expectations
        """

    @staticmethod
    def evaluate(metrics: Dict = None, params: Dict = None, experiment_path: str = ''):
        """Log metrics and parameters to mlflow


        :param metrics:             ETL pipeline metrics to be logged
        :param params:              ETL pipeline parameters to be logged
        :param experiment_path:     MLFlow Experiment path to log metrics
        """
        mlflow.set_tracking_uri(c.DATABRICKS_MLFLOW_URI)
        mlflow.set_experiment(experiment_path)
        mlflow.start_run()
        mlflow.log_metrics(metrics) if metrics else None
        mlflow.log_params(params) if params else None
        mlflow.end_run()

    def run(self):
        """Runs the ETL job

        """
        self.extract()
        self.transform()
        self.validate()
        self.load()
