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
TEST_FILEPATH_PREFIX = path.join(PROJECT_ROOT_DIRPATH, "data/processed/dataset.test_X")
PREDICT_FILENAME_PREFIX = "predict"
PREDICT_FILENAME_EXTENSION = "tsv"

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
KWARGS_TO_CSV = {
    "sep": "\t"
}


def gen_params_dict(n_estimators, max_depth, learning_rate,
                    reg_lambda, reg_alpha,
                    subsample, colsample_bytree, seed):
    return {"n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "seed": seed}


def get_test_X(filepath_prefix, location, fold_id=None):
    df = pd.read_csv('.'.join([filepath_prefix, location + ".tsv"]), **KWARGS_READ_CSV)

    if isinstance(fold_id, int):
        crossval_index_filename = '.'.join([filepath_prefix,
                                            "index.crossval{f}".format(f=fold_id),
                                            location + ".blp"])
        extract_index = pd.DatetimeIndex(bp.unpack_ndarray_file(crossval_index_filename))
        df = df.loc[extract_index, :]

        return df.iloc[:, :-1].values, df.index
    else:
        return df.values, df.index


@click.command()
@click.option("-t", "predict_target", flag_value="test", default=True)
@click.option("-v", "predict_target", flag_value="crossval")
@click.option("--location", "-l", type=str, default=None)
@click.option("--fold-id", "-f", type=int)
@click.option("--n_estimators", type=int, default=1000)
@click.option("--max_depth", type=int, default=3)
@click.option("--learning_rate", type=float, default=0.1)
@click.option("--reg_lambda", type=float, default=1.0)
@click.option("--reg_alpha", type=float, default=0.0)
@click.option("--subsample", type=float, default=0.8)
@click.option("--colsample_bytree", type=float, default=0.8)
@click.option("--seed", type=int, default=0)
def main(location, predict_target, fold_id,
         n_estimators, max_depth, learning_rate,
         reg_lambda, reg_alpha,
         subsample, colsample_bytree, seed):
    logger = logging.getLogger(__name__)
    logger.info('#0: run prediction ')

    #
    # predict by the serialized model
    #
    if location is None:
        location_list = LOCATIONS
    else:
        location_list = [location, ]

    XGB_PARAMS = gen_params_dict(n_estimators, max_depth, learning_rate,
                                 reg_lambda, reg_alpha,
                                 subsample, colsample_bytree, seed)
    param_str = str()
    for (key, value) in XGB_PARAMS.items():
        param_str += "{k}_{v}.".format(k=key, v=value)

    for place in location_list:
        if predict_target == "test":
            logger.info('#1: predict all training data by the model trained those @ {l} !'.format(l=place))

            m = MyXGBRegressor(model_name=param_str + "test.{l}".format(l=place), params=XGB_PARAMS)

            X_test, ret_index = get_test_X(TEST_FILEPATH_PREFIX, place, fold_id=None)
        elif predict_target == "crossval":
            if fold_id is None:
                raise ValueError("Specify validation dataset number as an integer !")

            logger.info('#1: predict test subset in cross-validation of fold-id: {f} @ {l} !'.format(f=fold_id, l=place))

            m = MyXGBRegressor(model_name=param_str + "crossval{i}.{l}".format(i=fold_id, l=place), params=XGB_PARAMS)

            X_test, ret_index = get_test_X(TRAIN_FILEPATH_PREFIX, place, fold_id=fold_id)
        else:
            raise ValueError("Invalid flag, '-t' or '-v' is permitted !")

        logger.info('#1: get test dataset @ {l} !'.format(l=place))

        logger.info('#2: now predicting...')
        y_pred = m.predict(X_test)


        pd.DataFrame(
            y_pred, index=ret_index, columns=[param_str[:-1]]
        ).to_csv(
            path.join(m.MODELS_SERIALIZING_BASEPATH,
                      '.'.join([PREDICT_FILENAME_PREFIX, m.model_name, PREDICT_FILENAME_EXTENSION])),
            **KWARGS_TO_CSV
        )

        logger.info('#2: a prediction result is saved @ {l} !'.format(l=place))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
