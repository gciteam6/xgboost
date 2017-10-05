# Third-party modules
import pandas as pd
# Hand-made modules
from base import DataFrameHandlerBase


KWARGS_READ_CSV = {
    "sep": "\t",
    "header": 0,
    "index_col": 0,
    "parse_dates": ["datetime"],
    "date_parser": lambda yyyymmddHHMM: pd.to_datetime(yyyymmddHHMM, format="%Y%m%d%H%S", errors="coerce"),
    "na_values": ['', '　']
}


class DataFrameHandler(DataFrameHandlerBase):
    def __init__(self):
        super().__init__()
        self.df_list = list()

    def set_tsvdata_of_every_10min(self, path_or_buf):
        self.df_list.append(pd.read_csv(path_or_buf, **KWARGS_READ_CSV))


if __name__ == '__main__':
    print("concatenation!")
