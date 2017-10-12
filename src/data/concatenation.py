# Built-in modules
from glob import glob
import re
# Third-party modules
import pandas as pd
# Hand-made modules
from .base import DataFrameHandlerBase

REGEX_FILE_EXTENSION = "data#5"
KWARGS_READ_CSV_INTERIM_FILE = {
    "index_col": 0
}


class DatasetCollector(DataFrameHandlerBase):
    def __init__(self, file_extension=REGEX_FILE_EXTENSION):
        super().__init__()
        self.file_extension = file_extension

    def read_tsv(self, path_or_buf):
        return pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_INTERIM_FILE))

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))

    def gen_filepath_prefix_suffix_nested_list(self, location_name):
        filepath_prefix_suffix_nested_list = list()
        pattern = re.compile("\.values\.")
        regex_values_filepath = self.path.join(
            self.INTERIM_DATA_BASEPATH,
            "*.values.{l}.data#[1-9]".format(l=location_name, e=self.file_extension)
        )

        for filepath in  glob(regex_values_filepath):
            match = pattern.search(filepath)
            filepath_prefix_suffix_nested_list.append(
                (filepath[:match.start()], filepath[match.end():])
            )

        return filepath_prefix_suffix_nested_list

    def retrieve_data(self, filepath_prefix_suffix_nested_list):
        if len(filepath_prefix_suffix_nested_list) < 1:
            raise ValueError("Empty ?")

        df_ret = self.read_blp_as_df(
            filepath_prefix_suffix_nested_list[0][0],
            filepath_prefix_suffix_nested_list[0][1]
        )

        if len(filepath_prefix_suffix_nested_list) > 1:
            for filepath_prefix_suffix in filepath_prefix_suffix_nested_list[1:]:
                df_ret = df_ret.merge(
                    self.read_blp_as_df(filepath_prefix_suffix[0],
                                        filepath_prefix_suffix[1]),
                    **self.KWARGS_OUTER_MERGE
                )

        return df_ret


if __name__ == '__main__':
    print("concatenation!")
