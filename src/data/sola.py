# Third-party modules
import pandas as pd
# Hand-made modules
from base import DataFrameHandlerBase

KWARGS_READ_CSV_SOLA = {
    "index_col": 0,
}
LABEL_SOLA_UKISHIMA = "SOLA01"
LABEL_SOLA_OUGISHIMA = "SOLA02"
LABEL_SOLA_YONEKURAYAMA = "SOLA03"


class SolarPhotovoltaicHandler(DataFrameHandlerBase):
    def __init__(self):
        super().__init__()
        self.LABEL_SOLA_UKISHIMA = LABEL_SOLA_UKISHIMA
        self.LABEL_SOLA_OUGISHIMA = LABEL_SOLA_OUGISHIMA
        self.LABEL_SOLA_YONEKURAYAMA = LABEL_SOLA_YONEKURAYAMA

    def read_tsv(self, path_or_buf):
        df_ret = pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_SOLA))
        df_ret.index = self.parse_datetime(pd.Series(df_ret.index).apply(str))
        return df_ret

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))


if __name__ == '__main__':
    print("Solar Photovoltaic !")
