# Built-in modules
from abc import abstractmethod
from copy import deepcopy
import csv
from os import pardir, path, makedirs
import pickle
# Third-party modules
import numpy as np
import bloscpack as bp
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT_PATH = path.join(path.dirname(__file__), pardir, pardir)
RAW_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/raw")
INTERIM_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/interim")
PROCESSED_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/processed")
MODELS_SERIALIZING_BASEPATH = path.join(PROJECT_ROOT_PATH, "models")


class PathHandlerBase(object):
    def __init__(self):
        self.PROJECT_ROOT_PATH = PROJECT_ROOT_PATH
        self.RAW_DATA_BASEPATH = RAW_DATA_BASEPATH
        self.INTERIM_DATA_BASEPATH = INTERIM_DATA_BASEPATH
        self.PROCESSED_DATA_BASEPATH = PROCESSED_DATA_BASEPATH
        self.MODELS_SERIALIZING_BASEPATH = MODELS_SERIALIZING_BASEPATH
        self.path = path

    @staticmethod
    def gen_abspath(relpath):
        abspath = path.abspath(relpath)
        makedirs(path.dirname(abspath), exist_ok=True)

        return abspath


class BloscpackMixin:
    @staticmethod
    def read_blp(serialized_filepath):
        return bp.unpack_ndarray_file(serialized_filepath)

    @staticmethod
    def to_blp(ndarray: np.array, serialized_filepath):
        bp.pack_ndarray_file(ndarray, serialized_filepath)

    @staticmethod
    def read_listfile(path_or_buf):
        with open(path_or_buf, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')

            return [string for row in reader for string in row]

    @staticmethod
    def to_listfile(string_list, path_or_buf):
        with open(path_or_buf, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(string_list)


class PickleMixin:
    @staticmethod
    def read_pkl(path_or_buf):
        with open(path_or_buf, mode="rb") as f:
            return pickle.load(f)

    @staticmethod
    def to_pkl(obj, path_or_buf):
        with open(path_or_buf, mode="wb") as f:
            pickle.dump(obj, f)


class MyEstimatorBase(PathHandlerBase):
    def __init__(self, model_name: str, params: dict):
        super().__init__()
        self.model_name = model_name
        self.params = params
        self.scorer = mean_absolute_error

    @abstractmethod
    def get_model_instance(self): pass

    @abstractmethod
    def fit(self, X, y): pass

    @abstractmethod
    def predict(self, X): pass

    def score(self, X, y):
        return self.scorer(y, self.predict(X))

    def gen_serialize_filepath(self, prefix, suffix):
        return '.'.join([prefix, self.model_name, suffix])

    def get_params(self, deep=True):
        return deepcopy(self.params) if deep else self.params

    def set_params(self, **params):
        for k, v in params.items():
            self.params[k] = v

        return self


def safe_indexing(X, index):
    """From 'safe_indexing' in sklearn.utils.base"""
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only index in pandas
        index = index if index.flags.writeable else index.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[index]
        except ValueError:
            return X.copy().iloc[index]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(index, 'dtype') and
                                           index.dtype.kind == 'i'):
            # This is often substantially faster than X[index]
            return X.take(index, axis=0)
        else:
            return X[index]
    else:
        return [X[idx] for idx in index]


def safe_split(X, y, index):
    """From '_safe_split' in sklearn.utils.metaestimators"""
    X_subset = safe_indexing(X, index)
    y_subset = safe_indexing(y, index)

    return X_subset, y_subset


if __name__ == '__main__':
    print("Here is src/models/base.py !")


