# Built-in modules
from os import path, pardir
import sys
import logging

# not used in this stub but often useful for finding various files
PROJECT_ROOT_DIRPATH = path.join(path.dirname(__file__), pardir, pardir)
sys.path.append(PROJECT_ROOT_DIRPATH)

# Third-party modules
import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
# Hand-made modules
from src.features.dataset import DatasetHandler
from src.features.dummy import DummyFeatureHandler
from src.features.time_series import TimeSeriesReshaper

FILE_EXTENTION = "tsv"
NAN_NUMBER_THRESHOLD = 10000
SHIFT_INDEX_OFFSETS_HOUR = -28
OBJECTIVE_LABEL_NAMES = ["kwh", ]

DROP_DATETIME_RANGE_UKISHIMA = [tuple()]
DROP_DATETIME_RANGE_OUGISHIMA = [tuple()]
DROP_DATETIME_RANGE_YONEKURAYAMA = [
    (pd.to_datetime("2012-01-01 00:10:00"), pd.to_datetime("2012-01-27 00:00:00")),
    (pd.to_datetime("2013-09-05 00:10:00"), pd.to_datetime("2015-03-07 00:00:00"))
]

LOCATIONS = (
    "ukishima",
    "ougishima",
    "yonekurayama",
)
CIRCULAR_CATEGORICAL_VARIABLES = (
    "wv",
)
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

    #
    # get samples from the serialized datasets
    #
    for location in LOCATIONS:
        dataset_filepath = path.join(
            maker.PROCESSED_DATA_BASEPATH,
            "dataset.amd_sfc_forecast_kwh.{l}.{e}".format(l=location, e=FILE_EXTENTION)
        )
        df_data_flags = maker.read_tsv(dataset_filepath)
        df_data, df_flags = maker.split_data_and_flags(df_data_flags)

        # TODO: amd, sfc利用フラグの扱いに関する処理の実装

        maker.to_blp_via_df(
            df_data,
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.data"),
            "{l}.features#1".format(l=location)
        )
        maker.to_blp_via_df(
            df_flags,
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.flags"),
            "{l}.features#1".format(l=location)
        )

        logger.info('#1: save data & flags as each files of {l} !'.format(l=location))
        del (df_data_flags, df_data, df_flags)

    logger.info('#1: end data-flags separations !')

    #
    # convert categorical to dummy
    #
    categ = DummyFeatureHandler()

    for location in LOCATIONS:
        df_data = maker.read_blp_as_df(
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.data"),
            "{l}.features#1".format(l=location)
        )

        logger.info('#2: read tsv file of {l} !'.format(l=location))

        for col_name, correspond_dict in categ.FORECAST_ATTRIBUTES.items():
            df_data[col_name] = categ.convert_series_along_dict(df_data[col_name], correspond_dict)

        sr_month = categ.extract_month(df_data.index)
        df_month_cos_sin = categ.convert_linear_to_circular(sr_month, categ.MONTH_CATEGORY_NUMBER)
        df_data = df_data.merge(df_month_cos_sin, **maker.KWARGS_OUTER_MERGE)

        sr_hour = categ.extract_hour(df_data.index)
        df_hour_cos_sin = categ.convert_linear_to_circular(sr_hour, categ.HOUR_CATEGORY_NUMBER)
        df_data = df_data.merge(df_hour_cos_sin, **maker.KWARGS_OUTER_MERGE)

        for col_name in CIRCULAR_CATEGORICAL_VARIABLES:
            df_temp_cos_sin = categ.convert_linear_to_circular(
                df_data[col_name], len(categ.FORECAST_ATTRIBUTES[col_name])
            )
            df_data = df_data.merge(df_temp_cos_sin, **maker.KWARGS_OUTER_MERGE)
            df_data.drop(col_name, axis=1, inplace=True)

        maker.to_blp_via_df(
            df_data,
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.data"),
            "{l}.features#2".format(l=location)
        )

        logger.info('#2: covert categorical features to dummy ones & save as a file in {l} !'.format(l=location))
        del (
            df_data,
            sr_month, df_month_cos_sin,
            sr_hour, df_hour_cos_sin
        )

    logger.info('#2: end categorical feature processing !')
    del categ

    #
    # nan processings (prune verbose features and fill nan)
    #
    reshaper = TimeSeriesReshaper()

    for location, drop_index_list in zip(LOCATIONS, DROP_DATETIME_RANGE_LOCATIONS):
        df_data = maker.read_blp_as_df(
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.data"),
            "{l}.features#2".format(l=location)
        )

        logger.info('#3: read tsv file of {l} !'.format(l=location))

        df_X, df_y = maker.separate_X_y(df_data)

        df_X = reshaper.drop_columns_of_many_nan(df_X, NAN_NUMBER_THRESHOLD)

        shift_col_name_list = reshaper.get_regex_matched_col_name(
            df_X.columns, reshaper.REGEX_SHIFT_COL_NAME_PREFIXES
        )
        df_X = reshaper.shift_indexes(
            df_X, pd.offsets.Hour(SHIFT_INDEX_OFFSETS_HOUR), shift_col_name_list
        )

        df_X.fillna(method="bfill", inplace=True)
        # TODO: より細かなfillna処理の実装(連続値のinterpolateによる補完, kwhの部分欠損値の補完)

        logger.info('#3: remove verbose features & fill nan in {l} !'.format(l=location))

        X_train, X_test = maker.separate_train_test(df_X)
        y_train, _ = maker.separate_train_test(df_y)
        df_train = X_train.merge(y_train, **maker.KWARGS_INNER_MERGE)

        for drop_index in drop_index_list:
            if len(drop_index) == 0:
                break

            print(reshaper.gen_datetime_index(drop_index[0], drop_index[1]))
            print(df_train)
            df_train.drop(
                reshaper.gen_datetime_index(drop_index[0], drop_index[1]),
                axis=0, inplace=True
            )

        maker.to_blp_via_df(
            df_data,
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.train"),
            "{l}.features#3".format(l=location)
        )
        maker.to_blp_via_df(
            df_data,
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.test"),
            "{l}.features#3".format(l=location)
        )

        logger.info('#3: save train/test dataset as each files in {l} !'.format(l=location))
        del (
            df_data, df_train, df_X, df_y,
            X_train, X_test,
            y_train,
            shift_col_name_list, drop_index_list
        )

    logger.info('#3: end nan processings !')
    del reshaper

    #
    # resample to every 30 min
    #
    for location in LOCATIONS:
        df_train_every_10 = maker.read_blp_as_df(
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.train"),
            "{l}.features#3".format(l=location)
        )

        logger.info('#4: read tsv file of the train section in {l} !'.format(l=location))

        X_train, y_train = maker.separate_X_y(df_train_every_10)
        X_train = X_train.resample(
            rule="30T", axis=0, closed="left"
        ).mean()
        y_train = y_train.resample(
            rule="30T", axis=0, closed="left"
        ).sum()
        df_train_every_30 = X_train.merge(y_train, **maker.KWARGS_INNER_MERGE)

        # TODO: 20 ~ 40 minから40 min, 30 ~ 50 minから50 minのsample生成

        logger.info('#4: convert sample frequency 10 min to 30 min of the train section in {l} !'.format(l=location))

        X_test = maker.read_blp_as_df(
            path.join(maker.INTERIM_DATA_BASEPATH, "dataset.test"),
            "{l}.features#3".format(l=location)
        )

        logger.info('#4: read tsv file of the test section in {l} !'.format(l=location))

        X_test = X_test.resample(
            rule="30T", axis=0, closed="left"
        ).mean()

        # TODO: 20 ~ 40 minから40 min, 30 ~ 50 minから50 minのsample生成

        logger.info('#4: convert sample frequency 10 min to 30 min of the test section in {l} !'.format(l=location))

        maker.to_blp_via_df(
            df_train_every_30,
            path.join(maker.PROCESSED_DATA_BASEPATH, "dataset.train_X_y"),
            "{l}.features#4".format(l=location)
        )
        maker.to_blp_via_df(
            X_test,
            path.join(maker.PROCESSED_DATA_BASEPATH, "dataset.test_X"),
            "{l}.features#4".format(l=location)
        )

        logger.info('#4: save train/test dataset as each files in {l} !'.format(l=location))
        del (df_train_every_10, df_train_every_30, X_train, X_test, y_train)

    logger.info('#4: end resampling !')
    del maker


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = path.join(path.dirname(__file__), pardir, pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
