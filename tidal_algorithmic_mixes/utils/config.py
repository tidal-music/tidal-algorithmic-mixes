import json
from datetime import datetime


class Config:
    def __init__(self, **kwargs):
        self.run_id = kwargs.get('run_id', None)
        self.git_hash = kwargs.get('git_hash', None)
        self.n_partitions = int(kwargs.get('n_partitions', 800))
        self.validate_kwargs(**kwargs)

    def validate_kwargs(self, **kwargs):
        """ Check that no unknown arguments are passed to the job. """
        for k in kwargs:
            if not hasattr(self, k):
                raise AttributeError('arg %s is not a member of %s' % (k, self.__class__.__name__))

    def as_dict(self, git_hash_prefix=None):
        """ Convert the config object to a dictionary with all members. """
        d = {}

        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                for h, val in v.items():
                    d[f"{k}_{h}"] = val
            elif isinstance(v, (bool, list)):
                d[k] = str(v)
            elif isinstance(v, (str, float, int)):
                d[k] = v
            elif isinstance(v, type(None)):
                d[k] = "None"

        if git_hash_prefix:
            ts = datetime.utcnow().replace(microsecond=0).isoformat()
            d[f"{git_hash_prefix}_git_hash_{ts}"] = d.pop('git_hash')

        return d

    @staticmethod
    def parse_boolean(kwarg):
        """ Parse a boolean kwarg. """
        if isinstance(kwarg, bool):
            return kwarg

        return json.loads(kwarg.lower())
