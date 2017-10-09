# Built-in modules
import re
# Third-party modules
import pandas as pd
# Hand-made modules
from .base import DataFrameHandlerBase, BloscpackMixin

OBJECTIVE_COLUMNS = ["kwh", ]
REGEX_FLAG_NAME_PREFIX = "f_"
REGEX_DROP_LABEL_NAME_PREFIXES = {
    "max_ws_",
    "ave_wv_",
    "ave_ws_",
    "max_tp_",
    "min_tp_",
    "sl_",
    "sd_",
    "vb_",
    "weather_",
    "dsr_",
    "dsd_",
    "dsr_"
}
DROP_LABEL_NAMES = [
    "weather",
    "weather_detail",
    "wind",
    "wave"
]


class DatasetHandler(DataFrameHandlerBase, BloscpackMixin):
    def __init__(self):
        super().__init__()
        self.OBJECTIVE_COLUMNS = OBJECTIVE_COLUMNS
        self.REGEX_FLAG_NAME_PREFIX = REGEX_FLAG_NAME_PREFIX
        self.REGEX_DROP_LABEL_NAME_PREFIXES = REGEX_DROP_LABEL_NAME_PREFIXES
        self.DROP_LABEL_NAMES = DROP_LABEL_NAMES

    def separate_train_test(self, df: pd.DataFrame):
        df_train = df.loc[self.TRAIN_DATE_RANGE[0]:self.TRAIN_DATE_RANGE[1], :]
        df_test = df.loc[self.TEST_DATE_RANGE[0]:self.TEST_DATE_RANGE[1], :]

        return df_train, df_test

    def separate_X_y(self, df: pd.DataFrame):
        df_y = df.loc[:, self.OBJECTIVE_COLUMNS].copy(deep=True)
        df.drop(self.OBJECTIVE_COLUMNS, axis=1, inplace=True)

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
