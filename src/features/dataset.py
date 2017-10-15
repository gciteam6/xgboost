# Built-in modules
from glob import glob
import re
# Third-party modules
import numpy as np
import pandas as pd
# Hand-made modules
from .base import DataFrameHandlerBase, BloscpackMixin

REGEX_FLAG_NAME_PREFIX = "f_"


class DatasetHandler(DataFrameHandlerBase, BloscpackMixin):
    def __init__(self, columns_y):
        super().__init__()
        self.columns_y = columns_y
        self.REGEX_FLAG_NAME_PREFIX = REGEX_FLAG_NAME_PREFIX

    def separate_train_test(self, df: pd.DataFrame):
        df_train = df.loc[self.TRAIN_DATE_RANGE[0]:self.TRAIN_DATE_RANGE[1], :]
        df_test = df.loc[self.TEST_DATE_RANGE[0]:self.TEST_DATE_RANGE[1], :]

        return df_train, df_test

    def separate_X_y(self, df: pd.DataFrame):
        df_y = df.loc[:, self.columns_y].copy(deep=True)
        df.drop(self.columns_y, axis=1, inplace=True)

        return df, df_y

    def split_data_and_flags(self, df):
        pattern = re.compile("^" + self.REGEX_FLAG_NAME_PREFIX)
        flag_col_name_list = [
            col_name for col_name in df.columns \
            if pattern.match(col_name)
        ]

        df_flags = df[flag_col_name_list].copy(deep=True)
        df.drop(flag_col_name_list, axis=1, inplace=True)

        return df, df_flags

    @staticmethod
    def get_regex_matched_col_name(col_name_list, regex_name_prefix_list):
        return [
            col_name \
            for col_name in col_name_list \
            for name_prefix in regex_name_prefix_list \
            if re.match("^" + name_prefix, col_name)
        ]

    def read_blp_as_df(self, prefix_filepath, suffix_filepath):
        values = self.read_blp(
            '.'.join([prefix_filepath, "values", suffix_filepath])
        )
        index = self.read_blp(
            '.'.join([prefix_filepath, "index", suffix_filepath])
        )
        columns = self.read_blp(
            '.'.join([prefix_filepath, "columns", suffix_filepath])
        )

        return pd.DataFrame(values, index=pd.DatetimeIndex(index), columns=columns)

    def to_blp_via_df(self, df, prefix_filepath, suffix_filepath):
        self.to_blp(
            df.values.astype('U8'),
            '.'.join([prefix_filepath, "values", suffix_filepath])
        )
        self.to_blp(
            np.asarray(df.index),
            '.'.join([prefix_filepath, "index", suffix_filepath])
        )
        self.to_blp(
            np.asarray(df.columns).astype('U'),
            '.'.join([prefix_filepath, "columns", suffix_filepath])
        )

    @staticmethod
    def gen_filepath_prefix_suffix_nested_list(regex_values_filepath):
        filepath_prefix_suffix_nested_list = list()
        pattern = re.compile("\.values\.")

        for filepath in glob(regex_values_filepath):
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
                df_ret = df_ret.append(
                    self.read_blp_as_df(filepath_prefix_suffix[0],
                                        filepath_prefix_suffix[1]),
                    verify_integrity=True
                )

        return df_ret


if __name__ == '__main__':
    print("dataset maker !")
