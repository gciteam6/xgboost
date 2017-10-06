# Built-in modules
from glob import glob
# Third-party modules
import pandas as pd
# Hand-made modules
from base import DataFrameHandlerBase

REGEX_FILE_EXTENSION = ".tsv"
KWARGS_READ_CSV_INTERIM_FILE = {
    "index_col": 0
}


class DatasetCollector(DataFrameHandlerBase):
    def __init__(self):
        super().__init__()
        self.REGEX_FILE_EXTENSION = REGEX_FILE_EXTENSION

    def read_tsv(self, path_or_buf):
        return pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_INTERIM_FILE))

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))

    def gen_filepath_list(self, location_name):
        regex_filepath = self.path.join(
            self.INTERIM_DATA_BASEPATH,
            "*.{l}".format(l=location_name) + self.REGEX_FILE_EXTENSION
        )

        return glob(regex_filepath)

    def retrieve_data(self, filepath_list):
        if len(filepath_list) < 1:
            raise ValueError("Empty list ?")

        df_ret = self.read_tsv(filepath_list[0])

        if len(filepath_list) > 1:
            for filepath in filepath_list[1:]:
                df_ret = df_ret.merge(
                    self.read_tsv(filepath),
                    how="outer",
                    left_index=True,
                    right_index=True
                )

        return df_ret


if __name__ == '__main__':
    print("concatenation!")
