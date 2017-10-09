# Built-in modules
from copy import deepcopy
from os import pardir, path, makedirs
import datetime
# Third-party modules
import numpy as np
import pandas as pd
import bloscpack as bp

PROJECT_ROOT_PATH = path.join(path.dirname(__file__), pardir, pardir)
RAW_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/raw")
INTERIM_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/interim")
PROCESSED_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/processed")

TRAIN_DATE_RANGE = (
    pd.to_datetime("2012-01-01 00:10:00"),
    pd.to_datetime("2016-01-01 00:00:00")
)
TEST_DATE_RANGE = (
    pd.to_datetime("2016-01-01 00:10:00"),
    pd.to_datetime("2017-04-01 00:00:00")
)
KWARGS_READ_CSV_BASE = {
    "sep": "\t",
    "header": 0,
    "index_col": 0
}
KWARGS_TO_CSV_BASE = {
    "sep": "\t"
}

MONTH_LABEL_NAME = "month"
HOUR_LABEL_NAME = "hour"


class PathHandlerBase(object):
    def __init__(self):
        self.PROJECT_ROOT_PATH = PROJECT_ROOT_PATH
        self.RAW_DATA_BASEPATH = RAW_DATA_BASEPATH
        self.INTERIM_DATA_BASEPATH = INTERIM_DATA_BASEPATH
        self.PROCESSED_DATA_BASEPATH = PROCESSED_DATA_BASEPATH
        self.path = path

    @staticmethod
    def gen_abspath(relpath):
        abspath = path.abspath(relpath)
        makedirs(path.dirname(abspath), exist_ok=True)

        return abspath


class DataFrameHandlerBase(PathHandlerBase):
    def __init__(self):
        super().__init__()
        self.TRAIN_DATE_RANGE = TRAIN_DATE_RANGE
        self.TEST_DATE_RANGE = TEST_DATE_RANGE
        self.KWARGS_READ_CSV_BASE = KWARGS_READ_CSV_BASE
        self.KWARGS_TO_CSV_BASE = KWARGS_TO_CSV_BASE

    def read_tsv(self, path_or_buf, **kwargs):
        df_ret = pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(kwargs))
        df_ret.index = pd.to_datetime(pd.Series(df_ret.index))

        return df_ret

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))

    def gen_read_csv_kwargs(self, kwargs_to_add: dict):
        ret_dict = deepcopy(self.KWARGS_READ_CSV_BASE)
        if kwargs_to_add is not None:
            ret_dict.update(kwargs_to_add)

        return ret_dict

    def gen_to_csv_kwargs(self, kwargs_to_add: dict):
        ret_dict = deepcopy(self.KWARGS_TO_CSV_BASE)
        if kwargs_to_add is not None:
            ret_dict.update(kwargs_to_add)

        return ret_dict


class BloscpackMixin:
    @staticmethod
    def read_blp(serialized_filepath):
        bp.unpack_ndarray_file(serialized_filepath)

    @staticmethod
    def to_blp(ndarray: np.array, serialized_filepath):
        bp.pack_ndarray_file(ndarray, serialized_filepath)

    @staticmethod
    def to_txt(string_list, path_or_buf):
        with open(path_or_buf, 'w') as f:
            f.write("\t".join(string_list))


class CategoricalHandlerBase(DataFrameHandlerBase):
    def __init__(self):
        super().__init__()
        self.MONTH_LABEL_NAME = MONTH_LABEL_NAME
        self.HOUR_LABEL_NAME = HOUR_LABEL_NAME

    def extract_month(self, sr):
        if not isinstance(sr, pd.Series):
            sr = pd.Series(sr)

        sr.index = sr

        return sr.apply(lambda elem: elem.month).rename(self.MONTH_LABEL_NAME)

    def extract_hour(self, sr):
        if not isinstance(sr, pd.Series):
            sr = pd.Series(sr)

        sr.index = sr

        return sr.apply(lambda elem: elem.hour).rename(self.HOUR_LABEL_NAME)


if __name__ == '__main__':
    print("Here is src/feature/base.py !")


