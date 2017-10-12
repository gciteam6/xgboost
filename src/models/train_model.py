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
import bloscpack as bp
# Hand-made modules
from src.models.xgb import MyXGBRegressor

TRAIN_FILEPATH_PREFIX = path.join(PROJECT_ROOT_DIRPATH, "data/interim/dataset.train")
TRAIN_FILEPATH_SUFFIX = "#3"
XGB_PARAMS = {
    "n_estimators": 300,
    "nthread": -1,
    "seed": 1
}


def get_train_X_y(train_filepath_prefix, train_filepath_suffix, fold_id=None):
    func_gen_filepath = lambda file_attr: '.'.join([
        train_filepath_prefix, file_attr, train_filepath_suffix
    ])

    train_X_y = bp.unpack_ndarray_file(func_gen_filepath("values"))

    if isinstance(fold_id, int):
        extract_index = bp.unpack_ndarray_file(
            func_gen_filepath("index.fold_{f}".format(f=fold_id))
        )
        return train_X_y[extract_index, :]
    else:
        return train_X_y


@click.command()
@click.argument("location", type=str)
@click.option("-t", "predict_target", flag_value="test", default=True)
@click.option("-v", "predict_target", flag_value="cv")
@click.option("--fold-id", "-f", type=int)
def main(location, predict_target, fold_id):
    logger = logging.getLogger(__name__)
    logger.info('#0: train models')

    # TODO: -tがCLAで与えられたとき, 全データでモデル構築を行う
    # TODO: python train_model.py -t -> models/xgb/fit_model.test.location.pkl

    # TODO: -vがCLAで与えられたとき, -fの数で交差検証サブセットを取得する
    # TODO: python train_model.py -v -f 0 -> models/xgb/fit_model.fold0.location.pkl

    m = MyXGBRegressor(model_name="{l}".format(l=location), params=XGB_PARAMS)

    if predict_target == "test":
        logger.info('#1: fit the model with all training dataset !')
        train_X_y = get_train_X_y(TRAIN_FILEPATH_PREFIX, location + '.' + TRAIN_FILEPATH_SUFFIX)

    elif predict_target == "cv":
        if fold_id is None:
            raise ValueError("Specify validation dataset number as an integer !")

        logger.info('#1: fit the model without fold-id: {f} !'.format(f=fold_id))
        train_X_y = get_train_X_y(TRAIN_FILEPATH_PREFIX, location + '.' + TRAIN_FILEPATH_SUFFIX, fold_id=fold_id)

    else:
        raise ValueError("Invalid flag, '-t' or '-v' is permitted !")

    logger.info('#1: get training dataset !')

    logger.info('#2: now modeling...')
    m.fit(train_X_y[:, :-1], train_X_y[:, -1])

    logger.info('#2: model is pickled as a file !')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
