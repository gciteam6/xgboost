# Built-in modules
from os import getcwd, path, pardir
import re
from glob import glob
# Third-party modules
import pandas as pd
# Hand-made modules
from .base import PathHandlerBase

PROJECT_ROOT_DIRPATH = path.join(path.dirname(__file__), pardir, pardir)
KWARGS_READ_CSV = {
    "sep": "\t",
    "header": 0,
    "parse_dates": [0],
    "index_col": 0
}
KWARGS_OUTER_MERGE = {
    "how": "outer",
    "left_index": True,
    "right_index": True
}

Y_TRUE_FILEPATH_PREFIX = path.join(PROJECT_ROOT_DIRPATH, "data", "processed", "dataset.train_y")
Y_TRUE_FILEPATH_SUFFIX = "tsv"
XGB_PREDICCT_FILEPATH_PREFIX = path.join(PROJECT_ROOT_DIRPATH, "models", "xgb", "predict")
XGB_PREDICCT_FILEPATH_SUFFIX = "tsv"

REGEX_XGB_COLUMN_NAME = ".*".join(["n_estimators", "seed_[0-9]"])
max_depth_conditions = range(3, 7)
learning_rate_conditions = [0.05, 0.1, 0.2, 0.3, 0.5]
subsample_comnditions = [0.67, 0.8, 1.0]
colsample_bytree_comnditions = [0.8, 0.9, 1.0]
seed_conditions = range(5)


def gen_date_index(predict_target):
    if predict_target == "test":
        return pd.date_range(
            pd.to_datetime("2016-01-01 00:30:00"),
            pd.to_datetime("2017-04-01 00:00:00"),
            freq=pd.offsets.Minute(30)
        )
    elif predict_target == "crossval":
        return pd.date_range(
            pd.to_datetime("2012-01-01 00:30:00"),
            pd.to_datetime("2016-01-01 00:00:00"),
            freq=pd.offsets.Minute(30)
        )
    else:
        raise ValueError("Invalid flag, 'test' or 'crossval' is permitted !")


def gen_xgb_experimental_condition_list():
    ret_list = [
        '.'.join([
            "n_estimators_{nest}".format(nest=nest),
            "max_depth_{mdep}".format(mdep=mdep),
            "learning_rate_{lrat}".format(lrat=lrat),
            "reg_lambda_{rlamb}".format(rlamb=rlamb),
            "reg_alpha_{ralp}".format(ralp=ralp),
            "subsample_{ssam}".format(ssam=ssam),
            "colsample_bytree_{csbt}".format(csbt=csbt),
            "seed_{seed}".format(seed=seed)
        ])
        for nest in [1000] \
        for mdep in max_depth_conditions \
        for lrat in learning_rate_conditions \
        for rlamb in [1.0] \
        for ralp in [0.0] \
        for ssam in subsample_comnditions \
        for csbt in colsample_bytree_comnditions \
        for seed in seed_conditions
    ]
    ret_list.extend([
        '.'.join([
            "n_estimators_{nest}".format(nest=nest),
            "max_depth_{mdep}".format(mdep=mdep),
            "learning_rate_{lrat}".format(lrat=lrat),
            "reg_lambda_{rlamb}".format(rlamb=rlamb),
            "reg_alpha_{ralp}".format(ralp=ralp),
            "subsample_{ssam}".format(ssam=ssam),
            "colsample_bytree_{csbt}".format(csbt=csbt),
            "seed_{seed}".format(seed=seed)
        ])
        for nest in [1000] \
        for mdep in max_depth_conditions \
        for lrat in learning_rate_conditions \
        for rlamb in [0.0] \
        for ralp in [1.0] \
        for ssam in subsample_comnditions \
        for csbt in colsample_bytree_comnditions \
        for seed in seed_conditions
    ])

    return ret_list


class MyStacker(PathHandlerBase):
    def __init__(self):
        super().__init__()
        self.KWARGS_READ_CSV = KWARGS_READ_CSV
        self.KWARGS_OUTER_MERGE = KWARGS_OUTER_MERGE
        self.REGEX_XGB_COLUMN_NAME = REGEX_XGB_COLUMN_NAME
        self.X_train_ = pd.DataFrame(None)

    @staticmethod
    def gen_y_true_filepath(location,
                            prefix=Y_TRUE_FILEPATH_PREFIX,
                            suffix=Y_TRUE_FILEPATH_SUFFIX):
        return '.'.join([prefix, location, suffix])

    @staticmethod
    def gen_xgb_predict_filepath_list(predict_target,
                                      location,
                                      prefix=XGB_PREDICCT_FILEPATH_PREFIX,
                                      suffix=XGB_PREDICCT_FILEPATH_SUFFIX):
        return glob('*.*'.join([prefix, predict_target, location, suffix]))

    @staticmethod
    def concat_prediction_results(df, filepath_list, regex_ignore_column=None):
        # Is the data in the specific column not contain NaN ?
        is_filled_column = ~df.isnull().any()

        if regex_ignore_column is None:
            for filepath in filepath_list:
                df_temp = pd.read_csv(filepath, **KWARGS_READ_CSV)
                df.loc[df_temp.index, df_temp.columns] = df_temp
                print(path.basename(filepath), "ended")
        else:
            pattern = re.compile(regex_ignore_column)
            for filepath in filepath_list:
                condition = path.basename(filepath)
                matcher = pattern.search(condition)
                if is_filled_column.loc[condition[matcher.start():matcher.end()]]:
                    print(condition, "passed")
                else:
                    df_temp = pd.read_csv(filepath, **KWARGS_READ_CSV)
                    df.loc[df_temp.index, df_temp.columns] = df_temp
                    print(condition, "extracted")

        return df

    def get_concatenated_xgb_predict(self, predict_target: str, location: str):
        if predict_target == "test" or predict_target == "crossval":
            xgb_predict_filepath_list = \
                self.gen_xgb_predict_filepath_list(predict_target, location)
        else:
            raise ValueError("Invalid flag, 'test' or 'crossval' is permitted !")

        try:
            df_pred = pd.read_csv(
                self.path.join(self.PROCESSED_DATA_BASEPATH,
                               "dataset.predict_y.layer_0.{t}.{l}.tsv".format(
                                   t=predict_target, l=location)),
                **KWARGS_READ_CSV
            )
        except FileNotFoundError:
            df_pred = pd.DataFrame(
                index=gen_date_index(predict_target),
                columns=gen_xgb_experimental_condition_list()
            )

        return self.concat_prediction_results(df_pred, xgb_predict_filepath_list,
                                              regex_ignore_column=self.REGEX_XGB_COLUMN_NAME)

    def get_concatenated_blending_predict(self,
                                          base_dirpath: str,
                                          num_layer: int,
                                          predict_target: str,
                                          location: str,
                                          suffix="tsv"):
        if predict_target == "test" or predict_target == "crossval":
            stacking_predict_filepath_list = \
                glob('.*'.join([base_dirpath, predict_target, location, suffix]))
        else:
            raise ValueError("Invalid flag, 'test' or 'crossval' is permitted !")

        try:
            df_pred = pd.read_csv(
                self.path.join(self.PROCESSED_DATA_BASEPATH,
                               "dataset.predict_y.layer_{n}.{t}.{l}.tsv".format(
                                   n=num_layer, t=predict_target, l=location
                               )),
                **KWARGS_READ_CSV
            )
        except FileNotFoundError:
            df_pred = pd.DataFrame(
                index=gen_date_index(predict_target),
            )

        return self.concat_prediction_results(df_pred, stacking_predict_filepath_list)


if __name__ == "__main__":
    print("My Stacking !")
