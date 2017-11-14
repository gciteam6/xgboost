# Built-in modules
from os import path, pardir
import sys
import logging
import re

# not used in this stub but often useful for finding various files
PROJECT_ROOT_DIRPATH = path.join(path.dirname(__file__), pardir, pardir)
sys.path.append(PROJECT_ROOT_DIRPATH)

# Third-party modules
import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ParameterGrid
# Hand-made modules
from src.models.blending import MyBlender
from src.models.stacking import MyStacker

BLEND_MODEL_INSTANCE = PLSRegression()
BLEND_MODEL_BASENAME = "layer1.PLSRegression"
BLEND_MODEL_SEARCHING_PARAMS = {"n_components": np.arange(1, 4)}

LOCATIONS = (
    "ukishima",
    "ougishima",
    "yonekurayama"
)
KWARGS_TO_CSV = {
    "sep": "\t"
}


def gen_params_grid():
    return ParameterGrid(BLEND_MODEL_SEARCHING_PARAMS)


def gen_param_string(param_dict):
    param_str = str()
    for (key, value) in param_dict.items():
        param_str += "{k}_{v}.".format(k=key, v=value)

    return param_str[:-1]


def gen_blender_list(predict_target, location):
    blend_model_param_list = gen_params_grid()
    blend_model_name_list = [
        BLEND_MODEL_BASENAME + \
        ".{p}.{t}.{l}".format(p=gen_param_string(blend_model_param), t=predict_target, l=location) \
        for blend_model_param in blend_model_param_list
    ]

    blender_list = [
        MyBlender(BLEND_MODEL_INSTANCE, blend_model_name, blend_model_param) \
        for blend_model_name, blend_model_param in zip(blend_model_name_list, blend_model_param_list)
    ]

    return blender_list


def remove_predict_target_and_location_suffix(target_string, predict_target):
    matcher = re.search(predict_target, target_string)

    return target_string[:matcher.start()-1]


def run_blending(predict_target, location, blender, stacker):
    logger = logging.getLogger(__name__)
    logger.info('#0: train models')

    # retrieve train y
    y_true_as_train = pd.read_csv(stacker.gen_y_true_filepath(location),
                                  **stacker.KWARGS_READ_CSV)
    y_true_as_train.dropna(axis=0, inplace=True)

    logger.info('#1: get y_true @ {l} !'.format(l=location))

    # retrieve train X
    df_pred_as_train = stacker.X_train_.loc[y_true_as_train.index, ~stacker.X_train_.isnull().any()]

    logger.info('#1: get y_pred as a train data @ {l} !'.format(l=location))

    #
    # bifurcation
    #
    if predict_target == "crossval":
        # try cross-validation
        pd.DataFrame(
            blender.cross_val_predict(df_pred_as_train.as_matrix(), y_true_as_train.as_matrix()),
            index=df_pred_as_train.index,
            columns=[remove_predict_target_and_location_suffix(blender.model_name, predict_target), ]
        ).to_csv(
            blender.gen_abspath(blender.gen_serialize_filepath("predict", "tsv")),
            **KWARGS_TO_CSV
        )

        logger.info('#2: estimate y_pred of train samples like cross-validation @ {l} !'.format(l=location))

    elif predict_target == "test":
        # fit model with the whole samples
        blender.fit(df_pred_as_train.as_matrix(), y_true_as_train.as_matrix())

        logger.info('#2: fit & serialized a model @ {l} !'.format(l=location))

        # retrieve test X
        df_pred_as_test = stacker.get_concatenated_xgb_predict(predict_target, location)
        df_pred_as_test.to_csv(
            stacker.path.join(stacker.PROCESSED_DATA_BASEPATH,
                              "dataset.predict_y.layer_0.{t}.{l}.tsv".format(t=predict_target, l=location)),
            **KWARGS_TO_CSV
        )

        logger.info('#3: get y_pred as a test data @ {l} !'.format(l=location))

        # predict
        pd.DataFrame(
            blender.predict(df_pred_as_test[df_pred_as_train.columns].as_matrix()),
            index=df_pred_as_test.index,
            columns=[remove_predict_target_and_location_suffix(blender.model_name, predict_target), ]
        ).to_csv(
            blender.gen_abspath(
                blender.gen_serialize_filepath("predict", "tsv")),
            **KWARGS_TO_CSV
        )

        logger.info('#4: estimate & save y_pred of test samples @ {l} !'.format(l=location))


@click.command()
@click.option("-t", "predict_target", flag_value="test", default=True)
@click.option("-v", "predict_target", flag_value="crossval")
@click.option("--location", "-l", type=str, default=None)
def main(predict_target, location):
    if location is None:
        location_list = LOCATIONS
    else:
        location_list = [location, ]

    for place in location_list:
        # get blender and stacker
        blender_list = gen_blender_list(predict_target, place)
        stacker = MyStacker()
        # attatch train X to the stacker
        stacker.X_train_ = stacker.get_concatenated_xgb_predict("crossval", place)
        stacker.X_train_.to_csv(
            path.join(stacker.PROCESSED_DATA_BASEPATH,
                      "dataset.predict_y.layer_0.crossval.{l}.tsv".format(l=location)),
            **KWARGS_TO_CSV
        )

        for blender in blender_list:
            run_blending(predict_target, place, blender, stacker)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
