# Built-in modules
import math
# Third-party modules
import pandas as pd
# Hand-made modules
from .base import CategoricalHandlerBase

MONTH_CATEGORY_NUMBER = 12
HOUR_CATEGORY_NUMBER = 24
COS_SIN_SUFFIXES = [
    "_cos",
    "_sin"
]

WEATHER_CORRESPOND_DICT = {
    "晴れ": 1,
    "くもり": 0,
    "雨": -1,
    "雪": -2
}
WIND_VECTOR_CORRESPOND_DICT = {
    "北東": 1,
    "北": 2,
    "北西": 3,
    "西": 4,
    "南西": 5,
    "南": 6,
    "南東": 7,
    "東": 8
}
WIND_INTENSITY_CORRESPOND_DICT = {
    "1.0": 1,  # 0～2m/s
    "2.0": 2,  # 3～5m/s
    "3.0": 3,  # 6～9m/s
    "4.0": 4   # 10m/s以上
}
FORECAST_ATTRIBUTES = {
    "we": WEATHER_CORRESPOND_DICT,
    "wv": WIND_VECTOR_CORRESPOND_DICT,
    "wc": WIND_INTENSITY_CORRESPOND_DICT
}


class DummyFeatureHandler(CategoricalHandlerBase):
    def __init__(self):
        super().__init__()
        self.COS_SIN_SUFFIXES = COS_SIN_SUFFIXES
        self.MONTH_CATEGORY_NUMBER = MONTH_CATEGORY_NUMBER
        self.HOUR_CATEGORY_NUMBER = HOUR_CATEGORY_NUMBER

        self.FORECAST_ATTRIBUTES = FORECAST_ATTRIBUTES

    def convert_linear_to_circular(self, sr: pd.Series, n_categories):
        col_name_list = [
            sr.name + suffix for suffix in self.COS_SIN_SUFFIXES
        ]
        sr_list = list()

        for func, col_name in zip([math.cos, math.sin], col_name_list):
            sr_list.append(
                sr.apply(lambda elem: func(2.0 * math.pi * elem / n_categories)).rename(col_name)
            )

        return pd.concat(sr_list, axis=1)

    @staticmethod
    def convert_series_along_dict(sr: pd.Series, correspond_dict):
        return sr.apply(lambda categorical_elem: correspond_dict[str(categorical_elem)]).rename(sr.name).astype(int)


if __name__ == '__main__':
    print("dummy feature !")
