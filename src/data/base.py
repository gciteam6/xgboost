# Built-in modules
from copy import deepcopy
from os import pardir, path, makedirs
import datetime
# Third-party modules
import pandas as pd

PROJECT_ROOT_PATH = path.join(path.dirname(__file__), pardir, pardir)
RAW_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/raw")
INTERIM_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/interim")
PROCESSED_DATA_BASEPATH = path.join(PROJECT_ROOT_PATH, "data/processed")

DATETIME_FORMAT = "(?P<year>\d{4})(?P<month>\d{1,2})(?P<day>\d{1,2})(?P<hour>\d{2})(?P<minute>\d{2})"
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
    "na_values": ['', '　']
}
KWARGS_TO_CSV_BASE = {
    "sep": "\t"
}
KWARGS_MERGE_TWO_DATAFRAME = {
    "how": "outer",
    "left_index": True,
    "right_index": True
}

LATLNGALT_UKISHIMA = (35.517558, 139.786920, 4.7)
LATLNGALT_OUGISHIMA = (35.488680, 139.727451, 4.8)
LATLNGALT_YONEKURAYAMA = (35.583302, 138.573118, 366.9)
LABEL_LAT_HOUR, LABEL_LAT_MINUTE = "lat1", "lat2"
LABEL_LNG_HOUR, LABEL_LNG_MINUTE = "lng1", "lng2"
LABEL_LAT_DECIMAL, LABEL_LNG_DECIMAL = "lat_dec", "lng_dec"


class PathHandlerBase(object):
    def __init__(self):
        self.PROJECT_ROOT_PATH = PROJECT_ROOT_PATH
        self.RAW_DATA_BASEPATH = RAW_DATA_BASEPATH
        self.INTERIM_DATA_BASEPATH = INTERIM_DATA_BASEPATH
        self.PROCESSED_DATA_BASEPATH = PROCESSED_DATA_BASEPATH
        self.KWARGS_MERGE_TWO_DATAFRAME = KWARGS_MERGE_TWO_DATAFRAME
        self.path = path

    @staticmethod
    def gen_abspath(relpath):
        abspath = path.abspath(relpath)
        makedirs(path.dirname(abspath), exist_ok=True)

        return abspath


class DataFrameHandlerBase(PathHandlerBase):
    def __init__(self):
        super().__init__()
        self.DATETIME_FORMAT = DATETIME_FORMAT
        self.TRAIN_DATE_RANGE = TRAIN_DATE_RANGE
        self.TEST_DATE_RANGE = TEST_DATE_RANGE
        self.KWARGS_READ_CSV_BASE = KWARGS_READ_CSV_BASE
        self.KWARGS_TO_CSV_BASE = KWARGS_TO_CSV_BASE

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

    @staticmethod
    def gen_datetime_index(start, end, freq_min: int = 10):
        return pd.date_range(start, end, freq=pd.offsets.Minute(freq_min))

    @staticmethod
    def gen_norm_datetime(year, month, day, hour, minute, second):
        return datetime.datetime(year, month, day) + \
               datetime.timedelta(hours=hour, minutes=minute, seconds=second)

    @staticmethod
    def add_annotations_to_column_names(df, attribute_name, location_name):
        return [
            '_'.join([
                str(column_name), attribute_name, location_name
            ]) for column_name in df.columns
        ]


class LocationHandlerBase(DataFrameHandlerBase):
    def __init__(self, master_filepath, **kwargs_location):
        super().__init__()
        self.LATLNGALT_UKISHIMA = LATLNGALT_UKISHIMA
        self.LATLNGALT_OUGISHIMA = LATLNGALT_OUGISHIMA
        self.LATLNGALT_YONEKURAYAMA = LATLNGALT_YONEKURAYAMA

        self.location = pd.read_csv(
            master_filepath, **self.gen_read_csv_kwargs(kwargs_location)
        )
        self.location[LABEL_LAT_DECIMAL] = self.location.apply(
            lambda df: self.cast_60_to_10(df[LABEL_LAT_HOUR], df[LABEL_LAT_MINUTE]), axis=1
        )
        self.location[LABEL_LNG_DECIMAL] = self.location.apply(
            lambda df: self.cast_60_to_10(df[LABEL_LNG_HOUR], df[LABEL_LNG_MINUTE]), axis=1
        )

    def get_near_observation_points(self, lat_mid, lng_mid, half_grid_size):
        lat_max, lat_min = lat_mid + half_grid_size, lat_mid - half_grid_size
        lng_max, lng_min = lng_mid + half_grid_size, lng_mid - half_grid_size

        lat_within_mesh = self.location[LABEL_LAT_DECIMAL].apply(
            lambda lat: True if (lat_min <= lat <= lat_max) else False
        )
        lng_within_mesh = self.location[LABEL_LNG_DECIMAL].apply(
            lambda lng: True if lng_min <= lng <= lng_max else False
        )

        flg_within_mesh = [is_lat and ls_lng for (is_lat, ls_lng) in zip(lat_within_mesh, lng_within_mesh)]

        return self.location.loc[flg_within_mesh, :]

    @staticmethod
    def cast_60_to_10(hour, minute, second=0):
        return hour + (minute / 60) + (second / 3600)


if __name__ == '__main__':
    print("Here is src/data/base.py !")
