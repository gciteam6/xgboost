# Third-party modules
import pandas as pd
# Hand-made modules
from .base import DataFrameHandlerBase

KWARGS_READ_CSV_SOLA = {
    "index_col": 0,
}
SOLA_LOCATION_LABEL_NAMES = {
    "ukishima": "SOLA01",
    "ougishima": "SOLA02",
    "yonekurayama": "SOLA03"
}


class SolarPhotovoltaicHandler(DataFrameHandlerBase):
    def __init__(self):
        super().__init__()
        self.SOLA_LOCATION_LABEL_NAMES = SOLA_LOCATION_LABEL_NAMES

    def read_tsv(self, path_or_buf):
        df_ret = pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_SOLA))
        df_ret.index = self.parse_datetime(pd.Series(df_ret.index).apply(str))
        return df_ret

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))


if __name__ == '__main__':
    print("Solar Photovoltaic !")
