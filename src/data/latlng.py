# Third-party modules
import pandas as pd
# Hand-made modules
from base import DataFrameHandlerBase


KWARGS_READ_CSV_AMD_MASTER = {
    "index_col": 0,
}

LATLNG_UKISHIMA = (35.517558, 139.786920, 4.7)
LATLNG_OUGIZIMA = (35.488680, 139.727451, 4.8)
LATLNG_YONEKURAYAMA = (35.583302, 138.573118, 366.9)
LABEL_LAT_HOUR, LABEL_LAT_MINUTE = "lat1", "lat2"
LABEL_LNG_HOUR, LABEL_LNG_MINUTE = "lng1", "lng2"


class AmedasPoint(DataFrameHandlerBase):
    def __init__(self, amd_master_filepath):
        super().__init__()

        self.df_amedas_point = pd.read_csv(
            amd_master_filepath, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_AMD_MASTER)
        )
        self.df_amedas_point["lat_dec"] = self.df_amedas_point.apply(
            lambda df: self.cast_60_to_10(df[LABEL_LAT_HOUR], df[LABEL_LAT_MINUTE]), axis=1
        )
        self.df_amedas_point["lng_dec"] = self.df_amedas_point.apply(
            lambda df: self.cast_60_to_10(df[LABEL_LNG_HOUR], df[LABEL_LNG_MINUTE]), axis=1
        )

    @staticmethod
    def cast_60_to_10(hour, minute, second=0):
        return hour + (minute / 60) + (second / 3600)


if __name__ == '__main__':
    print("concatenation!")
