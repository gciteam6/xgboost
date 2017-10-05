# Third-party modules
import pandas as pd
# Hand-made modules
from base import DataFrameHandlerBase


KWARGS_READ_CSV_AMD_MASTER = {
    "index_col": 0,
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


if __name__ == '__main__':
    print("concatenation!")
