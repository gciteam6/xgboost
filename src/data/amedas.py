from glob import glob
# Third-party modules
import pandas as pd
# Hand-made modules
from base import DataFrameHandlerBase


KWARGS_READ_CSV_AMD_MASTER = {
    "index_col": 0,
}
KWARGS_READ_CSV_AMD_LOG = {
    "index_col": 0,
    "na_values": ['', '　']
    # "parse_dates": ["datetime"],
    # "date_parser": lambda yyyymmddHHMM: pd.to_datetime(yyyymmddHHMM, format="%Y%m%d%H%M", errors="coerce"),
}

LATLNGALT_UKISHIMA = (35.517558, 139.786920, 4.7)
LATLNGALT_OUGIZIMA = (35.488680, 139.727451, 4.8)
LATLNGALT_YONEKURAYAMA = (35.583302, 138.573118, 366.9)
LABEL_LAT_HOUR, LABEL_LAT_MINUTE = "lat1", "lat2"
LABEL_LNG_HOUR, LABEL_LNG_MINUTE = "lng1", "lng2"
LABEL_LAT_DECIMAL, LABEL_LNG_DECIMAL = "lat_dec", "lng_dec"


class AmedasPoint(DataFrameHandlerBase):
    def __init__(self, amd_master_filepath):
        super().__init__()
        self.LATLNGALT_UKISHIMA = LATLNGALT_UKISHIMA
        self.LATLNGALT_OUGIZIMA = LATLNGALT_OUGIZIMA
        self.LATLNGALT_YONEKURAYAMA = LATLNGALT_YONEKURAYAMA

        self.df_amedas_point = pd.read_csv(
            amd_master_filepath, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_AMD_MASTER)
        )
        self.df_amedas_point[LABEL_LAT_DECIMAL] = self.df_amedas_point.apply(
            lambda df: self.cast_60_to_10(df[LABEL_LAT_HOUR], df[LABEL_LAT_MINUTE]), axis=1
        )
        self.df_amedas_point[LABEL_LNG_DECIMAL] = self.df_amedas_point.apply(
            lambda df: self.cast_60_to_10(df[LABEL_LNG_HOUR], df[LABEL_LNG_MINUTE]), axis=1
        )

    def get_near_amedas_points(self, lat_mid, lng_mid, half_grid_size):
        lat_max, lat_min = lat_mid + half_grid_size, lat_mid - half_grid_size
        lng_max, lng_min = lng_mid + half_grid_size, lng_mid - half_grid_size

        lat_within_mesh = self.df_amedas_point[LABEL_LAT_DECIMAL].apply(
            lambda lat: True if (lat_min <= lat <= lat_max) else False
        )
        lng_within_mesh = self.df_amedas_point[LABEL_LNG_DECIMAL].apply(
            lambda lng: True if lng_min <= lng <= lng_max else False
        )

        flg_within_mesh = [is_lat and ls_lng for (is_lat, ls_lng) in zip(lat_within_mesh, lng_within_mesh)]

        return self.df_amedas_point.loc[flg_within_mesh, :]

    @staticmethod
    def cast_60_to_10(hour, minute, second=0):
        return hour + (minute / 60) + (second / 3600)


class AmedasData(DataFrameHandlerBase):
    def __init__(self, amd_file_prefix="amd_", amd_file_suffix=".tsv"):
        super().__init__()
        self.amd_file_prefix = amd_file_prefix
        self.amd_file_suffix = amd_file_suffix
        self.amd_rexex_dirname = "amd[1-5]"

    def read_amd_data(self, path_or_buf):
        df_ret = pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_AMD_LOG))
        df_ret.index = self.parse_datetime(pd.Series(df_ret.index).apply(str))
        return df_ret

    def gen_filepath_list(self, aid_list):
        amd_regex_filepath_list = [
            self.path.join(
                self.INTERIM_DATA_BASEPATH,
                self.amd_rexex_dirname,
                self.amd_file_prefix + str(aid)) + self.amd_file_suffix \
            for aid in aid_list
        ]

        return [
            amd_file \
            for amd_regex_filepath in amd_regex_filepath_list \
            for amd_file in glob(amd_regex_filepath)
        ]

    def retrive_amedas_data(self, amd_filepath_list, amd_name_list):
        if len(amd_filepath_list) < 1:
            raise ValueError("Empty list ?")

        df_ret = self.read_amd_data(amd_filepath_list[0])
        df_ret.columns = [str(col_name) + '_' + amd_name_list[0] for col_name in df_ret.columns]

        if len(amd_filepath_list) > 1:
            for filepath, name in zip(amd_filepath_list[1:], amd_name_list[1:]):
                df_ret = df_ret.merge(
                    self.read_amd_data(filepath),
                    how="outer",
                    left_index=True,
                    right_index=True,
                    suffixes=(".", "_{}".format(name))
                )

        return df_ret

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))


if __name__ == '__main__':
    print("concatenation!")
