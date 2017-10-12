# Built-in modules
import re
# Third-party modules
import pandas as pd
# Hand-made modules
from .base import DataFrameHandlerBase

FORECAST_FILE_PREFIX = "forecast/forecast_"
FORECAST_FILE_SUFFIX = ".tsv"
KWARGS_READ_CSV_FORECAST_FILE = {
    "index_col": 0
}
FORECAST_REGIONS = {
    "ukishima": "kanagawa",
    "ougishima": "kanagawa",
    "yonekurayama": "yamanashi"
}
HHMMDD_BEGIN = "00:10:00"
REGEX_TIME_RANGED_NAME = "_\d{2}-\d{2}"


class ForecastHandler(DataFrameHandlerBase):
    def __init__(self):
        super().__init__()
        self.FORECAST_FILE_PREFIX = FORECAST_FILE_PREFIX
        self.FORECAST_FILE_SUFFIX = FORECAST_FILE_SUFFIX
        self.FORECAST_REGIONS = FORECAST_REGIONS
        self.HHMMDD_BEGIN = HHMMDD_BEGIN
        self.REGEX_TIME_RANGED_NAME = REGEX_TIME_RANGED_NAME

    def read_tsv(self, path_or_buf):
        df_ret = pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_FORECAST_FILE))
        df_ret.index = pd.to_datetime(pd.Series(df_ret.index + " " + self.HHMMDD_BEGIN))

        return df_ret

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))

    def gen_filepath(self, location_name):
        filename = self.FORECAST_FILE_PREFIX + \
                   self.FORECAST_REGIONS[location_name] + \
                   self.FORECAST_FILE_SUFFIX

        return self.path.join(self.INTERIM_DATA_BASEPATH, filename)

    def gen_filepath_list(self, location_name):
        return list(self.gen_filepath(location_name))

    def retrieve_data(self, filepath_list):
        if len(filepath_list) < 1:
            raise ValueError("Empty list ?")

        df_ret = self.read_tsv(filepath_list[0])

        if len(filepath_list) > 1:
            for filepath in filepath_list[1:]:
                df_ret = df_ret.merge(
                    self.read_tsv(filepath), **self.KWARGS_OUTER_MERGE
                )

        return df_ret

    def add_datetime_ticks(self, df):
        dt_index = self.gen_datetime_index(
            start=self.TRAIN_DATE_RANGE[0], end=self.TEST_DATE_RANGE[1]
        )

        df_ret = pd.DataFrame(index=dt_index, columns=df.columns)
        df_ret.loc[df.index, :] = df

        return df_ret

    def expand_whole_day_data(self, sr: pd.Series):
        appended_data = list()

        for (current_dt, val) in sr.iteritems():
            year, month, day = current_dt.year, current_dt.month, current_dt.day
            dt_index = self.gen_datetime_index(
                start=self.gen_norm_datetime(year, month, day, 0, 10, 0),
                end=self.gen_norm_datetime(year, month, day, 24, 0, 0)
            )

            appended_data.append(
                pd.Series([val for _ in range(dt_index.size)], index=dt_index, name=sr.name)
            )

        return pd.concat(appended_data, axis=0)

    def get_whole_day_data_columns(self, col_name_list):
        pattern = re.compile("^.*" + self.REGEX_TIME_RANGED_NAME + "$")

        return [col_name for col_name in col_name_list if not pattern.match(col_name)]

    def expand_time_ranged_data(self, sr: pd.Series):
        col_name = sr.name
        appended_data = list()

        for (current_dt, val) in sr.iteritems():
            year, month, day = current_dt.year, current_dt.month, current_dt.day
            dt_index = self.gen_datetime_index(
                start=self.gen_norm_datetime(year, month, day, int(col_name[-5:-3]), 10, 0),
                end=self.gen_norm_datetime(year, month, day, int(col_name[-2:]), 0, 0)
            )

            appended_data.append(pd.Series(
                [val for _ in range(dt_index.size)], index=dt_index, name=col_name)
            )

        return pd.concat(appended_data, axis=0)

    def get_time_ranged_data_columns(self, col_name_list):
        pattern = re.compile("^.*" + self.REGEX_TIME_RANGED_NAME + "$")

        return [col_name for col_name in col_name_list if pattern.match(col_name)]

    def extract_attribute_from_time_ranged_column_name(self, col_name):
        pattern = re.compile(self.REGEX_TIME_RANGED_NAME)
        match = re.search(pattern, col_name)

        return col_name[:match.start()]


if __name__ == '__main__':
    print("forecast!")
