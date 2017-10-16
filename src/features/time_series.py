# Built-in modules
import re
# Hand-made modules
from .base import DataFrameHandlerBase
REGEX_DROP_LABEL_NAME_PREFIXES = [
    "weather_"
]
DROP_LABEL_NAMES = [
    "weather",
    "weather_detail",
    "wind",
    "wave"
]
REGEX_SHIFT_COL_NAME_PREFIXES = [
    "pr_",
    "max_iws_",
    "ave_wv_",
    "ave_ws_",
    "max_tp_",
    "min_tp_",
    "sl_",
    "dsr_",
    "gsr_",
    "sd_",
    "dsd_",
    "lap_",
    "sap_",
    "cap_",
    "3h_cap_",
    "rhm_",
    "min_rhm_",
    "vp_",
    "dtp_",
]


class TimeSeriesReshaper(DataFrameHandlerBase):
    def __init__(self):
        super().__init__()
        self.REGEX_DROP_LABEL_NAME_PREFIXES = REGEX_DROP_LABEL_NAME_PREFIXES
        self.DROP_LABEL_NAMES = DROP_LABEL_NAMES
        self.REGEX_SHIFT_COL_NAME_PREFIXES = REGEX_SHIFT_COL_NAME_PREFIXES

    def shift_indexes(self, df, freq, shift_col_name_list):
        non_shift_col_name_list = [
            col_name for col_name in df.columns \
            if col_name not in shift_col_name_list
        ]

        df_shifted = df[shift_col_name_list].shift(freq=freq, axis=0)
        df_non_shifted = df[non_shift_col_name_list]

        return df_non_shifted.merge(df_shifted, **self.KWARGS_OUTER_MERGE)

    @staticmethod
    def get_regex_matched_col_name(col_name_list, regex_name_prefix_list):
        return [
            col_name \
            for col_name in col_name_list \
            for name_prefix in regex_name_prefix_list \
            if re.compile("^" + name_prefix + ".*$").match(col_name)
        ]

    @staticmethod
    def drop_columns_of_many_nan(df, nan_number_threshold):
        reserve_column_list = [
            v < nan_number_threshold for v in df.isnull().sum(axis=0)
        ]

        return df.loc[:, reserve_column_list]


if __name__ == '__main__':
    print("time series !")
