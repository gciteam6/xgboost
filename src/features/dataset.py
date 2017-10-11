# Built-in modules
import re
# Third-party modules
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
        pattern = re.compile("^" + self.REGEX_FLAG_NAME_PREFIX + ".*$")
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
            if re.compile("^" + name_prefix + ".*$").match(col_name)
        ]


if __name__ == '__main__':
    print("dataset maker !")
