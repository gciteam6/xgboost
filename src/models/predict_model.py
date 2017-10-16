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

TRAIN_FILEPATH_PREFIX = path.join(PROJECT_ROOT_DIRPATH, "data/processed/dataset.test_X")
TRAIN_FILEPATH_EXTENTION = "blp"
XGB_PARAMS = {
    "n_estimators": 500,
    "nthread": -1,
    "seed": 1
}
LOCATIONS = (
    "ukishima",
    "ougishima",
    "yonekurayama"
)


def get_test_X(train_filepath_prefix, train_filepath_suffix, fold_id=None):
    func_gen_filepath = lambda file_attr: '.'.join([train_filepath_prefix,
                                                    file_attr,
                                                    train_filepath_suffix])
    values = bp.unpack_ndarray_file(func_gen_filepath("values"))
    if isinstance(fold_id, int):
        whole_index = bp.unpack_ndarray_file(func_gen_filepath("index"))
        columns = bp.unpack_ndarray_file(func_gen_filepath("columns"))
        df_train = pd.DataFrame(values, index=pd.DatetimeIndex(whole_index), columns=columns)

        removal_index = pd.DatetimeIndex(
            bp.unpack_ndarray_file(func_gen_filepath("crossval{f}".format(f=fold_id)))
        )
        df_train.drop(removal_index, axis=0, inplace=True)

        return df_train.values.astype(float)
    else:
        return values.astype(float)


@click.command()
@click.option("-t", "predict_target", flag_value="test", default=True)
@click.option("-v", "predict_target", flag_value="crossval")
@click.option("--location", "-l", type=str, default=None)
@click.option("--fold-id", "-f", type=int)
def main(location, predict_target, fold_id):
    logger = logging.getLogger(__name__)
    logger.info('#0: train models')

    #
    # predict by the serialized model
    #
    if location is None:
        location_list = LOCATIONS
    else:
        location_list = [location, ]

    for place in location_list:
        if predict_target == "test":
            logger.info('#1: predict all training data by the model trained those @ {l} !'.format(l=place))

            m = MyXGBRegressor(model_name="test.{l}".format(l=place), params=XGB_PARAMS)

            X_test = get_test_X(TRAIN_FILEPATH_PREFIX,
                                place + '.' + TRAIN_FILEPATH_EXTENTION,
                                fold_id=None)
        elif predict_target == "crossval":
            if fold_id is None:
                raise ValueError("Specify validation dataset number as an integer !")

            logger.info('#1: predict test subset in cross-validation of fold-id: {f} @ {l} !'.format(f=fold_id, l=place))

            m = MyXGBRegressor(model_name="crossval{i}.{l}".format(i=fold_id, l=place), params=XGB_PARAMS)

            X_test = get_test_X(TRAIN_FILEPATH_PREFIX,
                                place + '.' + TRAIN_FILEPATH_EXTENTION,
                                fold_id=fold_id)
        else:
            raise ValueError("Invalid flag, '-t' or '-v' is permitted !")

        logger.info('#1: get test dataset @ {l} !'.format(l=place))

        logger.info('#2: now predicting...')
        y_pred = m.predict(X_test)

        m.to_blp(
            y_pred,
            path.join([m.MODELS_SERIALIZING_BASEPATH,
                       '.'.join(["predict", m.model_name, "{l}.blp".format(l=place)])
                       ])
        )

        logger.info('#2: a prediction result is saved @ {l} !'.format(l=place))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
