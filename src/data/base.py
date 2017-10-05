# Built-in modules
from copy import deepcopy
from os import pardir, path, makedirs
# Third-party modules
import pandas as pd


PROJECT_ROOT_PATH = path.join(path.dirname(__file__), pardir, pardir)
RAW_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/raw")
INTERIM_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/interim")
PROCESSED_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/processed")

DATETIME_FORMAT = "(?P<year>\d{4})(?P<month>\d{1,2})(?P<day>\d{1,2})(?P<hour>\d{2})(?P<minute>\d{2})"
KWARGS_READ_CSV_BASE = {
    "sep": "\t",
    "header": 0,
    "na_values": ['', '　']
}
KWARGS_TO_CSV_BASE = {
    "sep": "\t"
}


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
        self.KWARGS_READ_CSV_BASE = KWARGS_READ_CSV_BASE
        self.KWARGS_TO_CSV_BASE = KWARGS_TO_CSV_BASE
        self.DATETIME_FORMAT = DATETIME_FORMAT

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

    def parse_datetime(self, df):
        return pd.to_datetime(df.str.extract(self.DATETIME_FORMAT, expand=False))


if __name__ == '__main__':
    print("Here is src/data/base.py !")
