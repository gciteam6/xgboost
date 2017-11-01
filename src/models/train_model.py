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
import bloscpack as bp
# Hand-made modules
from src.models.xgb import MyXGBRegressor

TRAIN_FILEPATH_PREFIX = path.join(PROJECT_ROOT_DIRPATH, "data/processed/dataset.train_X_y")
TRAIN_FILEPATH_EXTENTION = "tsv"

LOCATIONS = (
    "ukishima",
    "ougishima",
    "yonekurayama"
)
KWARGS_READ_CSV = {
    "sep": "\t",
    "header": 0,
    "parse_dates": [0],
    "index_col": 0
}


def gen_params_dict(n_estimators, max_depth, learning_rate,
                    penal_lambda, penal_alpha,
                    subsample, colsample_bytree, seed):
    return {"n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "lambda": penal_lambda,
            "alpha": penal_alpha,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "seed": seed}


def get_train_X_y(train_filepath_prefix, train_filepath_suffix, fold_id=None):
    df_train = pd.read_csv('.'.join([train_filepath_prefix, train_filepath_suffix]),
                           **KWARGS_READ_CSV)

    if isinstance(fold_id, int):
        crossval_index_filename = '.'.join([train_filepath_prefix,
                                            "crossval{f}".format(f=fold_id),
                                            train_filepath_suffix])
        removal_index = pd.DatetimeIndex(bp.unpack_ndarray_file(crossval_index_filename))
        df_train.drop(removal_index, axis=0, inplace=True)

    df_train.dropna(axis=0, how="any", inplace=True)

    return df_train.values


@click.command()
@click.option("-t", "predict_target", flag_value="test", default=True)
@click.option("-v", "predict_target", flag_value="crossval")
@click.option("--location", "-l", type=str, default=None)
@click.option("--fold-id", "-f", type=int)
def main(location, predict_target, fold_id):
    logger = logging.getLogger(__name__)
    logger.info('#0: train models')

    #
    # fit the model
    #
    if location is None:
        location_list = LOCATIONS
    else:
        location_list = [location, ]

    for place in location_list:
        if predict_target == "test":
            logger.info('#1: fit the model with all training dataset @ {l} !'.format(l=place))

            train_X_y = get_train_X_y(TRAIN_FILEPATH_PREFIX,
                                      place + '.' + TRAIN_FILEPATH_EXTENTION,
                                      fold_id=None)
            m = MyXGBRegressor(model_name="test.{l}".format(l=place), params=XGB_PARAMS)
        elif predict_target == "crossval":
            if fold_id is None:
                raise ValueError("Specify validation dataset number as an integer !")

            logger.info('#1: fit the model without fold-id: {f} @ {l} !'.format(f=fold_id, l=place))

            train_X_y = get_train_X_y(TRAIN_FILEPATH_PREFIX,
                                      place + '.' + TRAIN_FILEPATH_EXTENTION,
                                      fold_id=fold_id)
            m = MyXGBRegressor(model_name="crossval{i}.{l}".format(i=fold_id, l=place), params=XGB_PARAMS)
        else:
            raise ValueError("Invalid flag, '-t' or '-v' is permitted !")

        logger.info('#1: get training dataset @ {l} !'.format(l=place))

        logger.info('#2: now modeling...')
        m.fit(train_X_y[:, :-1], train_X_y[:, -1])

        logger.info('#2: model is pickled as a file @ {l} !'.format(l=place))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
