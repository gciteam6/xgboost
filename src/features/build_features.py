# Built-in modules
import gc
from os import path, pardir
import sys
import logging

# not used in this stub but often useful for finding various files
PROJECT_ROOT_DIRPATH = path.join(path.dirname(__file__), pardir, pardir)
sys.path.append(PROJECT_ROOT_DIRPATH)

# Third-party modules
import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
# Hand-made modules
from src.features.dataset import DatasetHandler
from src.features.dummy import DummyFeatureHandler
from src.features.time_series import TimeSeriesReshaper

NAN_NUMBER_THRESHOLD = 10000

SHIFT_INDEX_OFFSET_HOURS = [28, 32, 36, 40, 44, 48]
OBJECTIVE_LABEL_NAMES = ["kwh", ]
TRAIN_SAMPLE_SETTINGS = ["train{i}".format(i=n_iter) for n_iter in range(8)]
TEST_SAMPLE_SETTINGS = ["test{i}".format(i=n_iter) for n_iter in range(3)]
SETTINGS = TRAIN_SAMPLE_SETTINGS + TEST_SAMPLE_SETTINGS
CIRCULAR_CATEGORICAL_VARIABLES = (
    "wv",
)
KWARGS_RESAMPLING = {
    "rule": "30T",
    "axis": 0,
    "closed": "left"
}

LOCATIONS = (
    "ukishima",
    "ougishima",
    "yonekurayama",
)
DROP_DATETIME_RANGE_UKISHIMA = [tuple()]
DROP_DATETIME_RANGE_OUGISHIMA = [tuple()]
DROP_DATETIME_RANGE_YONEKURAYAMA = [
    pd.date_range("2012-01-01 00:30:00", "2012-01-27 00:00:00", freq=pd.offsets.Minute(30)),
    pd.date_range("2013-09-05 00:30:00", "2015-03-07 00:00:00", freq=pd.offsets.Minute(30))
]
DROP_DATETIME_RANGE_LOCATIONS = (
    DROP_DATETIME_RANGE_UKISHIMA,
    DROP_DATETIME_RANGE_OUGISHIMA,
    DROP_DATETIME_RANGE_YONEKURAYAMA,
)


@click.command()
def main():
    logger = logging.getLogger(__name__)
    logger.info('#0: building features (explanatory values) from data')

    maker = DatasetHandler(columns_y=OBJECTIVE_LABEL_NAMES)

    # #
    # # convert categorical to dummy
    # #
    # categ = DummyFeatureHandler()
    #
    # for location in LOCATIONS:
    #     for setting in SETTINGS:
    #         df_data = maker.read_blp_as_df(
    #             path.join(maker.INTERIM_DATA_BASEPATH, "dataset.amd_sfc_forecast_kwh.{s}".format(s=setting)),
    #             "{l}.blp".format(l=location)
    #         )
    #
    #         logger.info('#1: load dataset to the memory @ {l}-{s} !'.format(l=location, s=setting))
    #
    #         for col_name, correspond_dict in categ.FORECAST_ATTRIBUTES.items():
    #             df_data[col_name] = categ.convert_series_along_dict(df_data[col_name], correspond_dict)
    #
    #         sr_month = categ.extract_month(df_data.index)
    #         df_month_cos_sin = categ.convert_linear_to_circular(sr_month, categ.MONTH_CATEGORY_NUMBER)
    #         df_data = df_data.merge(df_month_cos_sin, **maker.KWARGS_OUTER_MERGE)
    #
    #         sr_hour = categ.extract_hour(df_data.index)
    #         df_hour_cos_sin = categ.convert_linear_to_circular(sr_hour, categ.HOUR_CATEGORY_NUMBER)
    #         df_data = df_data.merge(df_hour_cos_sin, **maker.KWARGS_OUTER_MERGE)
    #
    #         for col_name in CIRCULAR_CATEGORICAL_VARIABLES:
    #             df_temp_cos_sin = categ.convert_linear_to_circular(
    #                 df_data[col_name], len(categ.FORECAST_ATTRIBUTES[col_name])
    #             )
    #             df_data = df_data.merge(df_temp_cos_sin, **maker.KWARGS_OUTER_MERGE)
    #             df_data.drop(col_name, axis=1, inplace=True)
    #
    #         maker.to_blp_via_df(
    #             df_data,
    #             path.join(maker.INTERIM_DATA_BASEPATH, "dataset.data.{s}".format(s=setting)),
    #             "{l}.features#1".format(l=location)
    #         )
    #
    #         logger.info('#1: covert categorical features to dummy ones & save as a file @ {l}-{s} !'.format(l=location, s=setting))
    #         del (
    #             df_data,
    #             sr_month, df_month_cos_sin,
    #             sr_hour, df_hour_cos_sin
    #         )
    #
    # logger.info('#1: end categorical feature processing !')
    # del categ
    # gc.collect()

    #
    # nan processings (prune verbose features and fill nan)
    #
    reshaper = TimeSeriesReshaper()

    for location, drop_index_list in zip(LOCATIONS, DROP_DATETIME_RANGE_LOCATIONS):
        filepath_prefix_suffix_nested_list = \
            maker.gen_filepath_prefix_suffix_nested_list(
                path.join(maker.INTERIM_DATA_BASEPATH,
                          "*.train[0-9].values.{l}.features#1".format(l=location))
            )
        filepath_prefix_suffix_nested_list.extend(
            maker.gen_filepath_prefix_suffix_nested_list(
                path.join(maker.INTERIM_DATA_BASEPATH,
                          "*.test[0-9].values.{l}.features#1".format(l=location))
            )
        )
        df_data = maker.retrieve_data(filepath_prefix_suffix_nested_list)

        logger.info('#2: load dataset to the memory @ {l} !'.format(l=location))

        for drop_index in drop_index_list:
            if len(drop_index) == 0:
                break
            else:
                df_data.drop(drop_index, axis=0, inplace=True)

        logger.info('#2: remove the specified index from df_X of {l} !'.format(l=location))

        df_X, df_y = maker.separate_X_y(df_data)
        df_X.replace("nan", np.nan, inplace=True)

        logger.info('#2: cast string nan to np.nan in df_X of {l} !'.format(l=location))

        drop_col_name_list = reshaper.get_regex_matched_col_name(
            df_data.columns, reshaper.REGEX_DROP_LABEL_NAME_PREFIXES
        )
        drop_col_name_list.extend(reshaper.DROP_LABEL_NAMES)
        df_X.drop(drop_col_name_list, axis=1, inplace=True)

        df_X = reshaper.drop_columns_of_many_nan(df_X, NAN_NUMBER_THRESHOLD)

        logger.info('#2: remove verbose features from df_X of {l} !'.format(l=location))

        shift_col_name_list = reshaper.get_regex_matched_col_name(
            df_X.columns, reshaper.REGEX_SHIFT_COL_NAME_PREFIXES
        )
        non_shift_col_name_list = [
            col_name for col_name in df_X.columns \
            if col_name not in shift_col_name_list
        ]

        df_shift = df_X[shift_col_name_list]
        df_X = df_X[non_shift_col_name_list]

        for shift_hour in SHIFT_INDEX_OFFSET_HOURS:
            df_temp = reshaper.get_shifted_dataframe(df_shift, pd.offsets.Hour(shift_hour))
            df_temp.columns = [col_name + "_{h}".format(h=shift_hour) \
                               for col_name in shift_col_name_list]
            df_X = df_X.merge(df_temp, **reshaper.KWARGS_OUTER_MERGE)

        logger.info('#2: shift the required data up @ {l} !'.format(l=location))

        df_X.fillna(method="bfill", inplace=True)
        df_data = df_X.merge(df_y, **maker.KWARGS_INNER_MERGE)

        logger.info('#2: fill nan in df_X of {l} !'.format(l=location))
        del (df_X, df_y,
             df_shift, shift_col_name_list)
        gc.collect()

        maker.to_blp_via_df(
            df_data,
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.data"),
            "{l}.features#2".format(l=location)
        )

        logger.info('#2: save dataset as a file @ {l} !'.format(l=location))
        del df_data
        gc.collect()

    logger.info('#2: end nan processings !')
    del reshaper
    gc.collect()

    #
    # resample to every 30 min
    #x
    for location in LOCATIONS:
        df_every_10 = maker.read_blp_as_df(
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.data"),
            "{l}.features#2".format(l=location)
        )

        logger.info('#3: load dataset to the memory @ {l} !'.format(l=location))

        df_every_10 = df_every_10.apply(pd.to_numeric, errors="coerce")

        logger.info('#3: cast string to numeric in dataset of {l} !'.format(l=location))

        df_X, df_y = maker.separate_X_y(df_every_10)
        df_X = df_X.resample(**KWARGS_RESAMPLING).mean()
        df_y = df_y.resample(**KWARGS_RESAMPLING).sum()
        df_every_30 = df_X.merge(df_y, **maker.KWARGS_INNER_MERGE)

        # TODO: 20 ~ 40 minから40 min, 30 ~ 50 minから50 minのsample生成

        logger.info('#3: convert sample frequency 10 min to 30 min of the train section in {l} !'.format(l=location))
        del (df_every_10, df_X, df_y)
        gc.collect()

        df_train, _ = maker.separate_train_test(df_every_30)
        maker.to_tsv(
            df_train,
            path.join(maker.PROCESSED_DATA_BASEPATH,
                      "dataset.train_X_y.{l}.tsv".format(l=location))
        )

        logger.info('#3: save train dataset as a file @ {l} !'.format(l=location))
        del df_train
        gc.collect()

        _, df_test = maker.separate_train_test(df_every_30)
        df_test, _ = maker.separate_X_y(df_test)
        maker.to_tsv(
            df_test,
            path.join(maker.PROCESSED_DATA_BASEPATH,
                      "dataset.test_X.{l}.tsv".format(l=location))
        )

        logger.info('#3: save test dataset as a file @ {l} !'.format(l=location))
        del (df_test, df_every_30)
        gc.collect()

    logger.info('#3: end resampling !')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
