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
from sklearn.cross_decomposition import PLSRegression
# Hand-made modules
from src.models.blending import MyBlender
from src.models.stacking import MyStacker

DATETIME_FORMAT = "(?P<year>\d{4})(?P<month>\d{1,2})(?P<day>\d{1,2})(?P<hour>\d{2})(?P<minute>\d{2})"
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


def gen_blender_and_stacker():
    blend_model_instance = PLSRegression()
    blend_model_params = {"n_components": 2}
    blend_model_name = "layer1.PLSRegression.n_components_2"

    return MyBlender(blend_model_instance, blend_model_name, blend_model_params), MyStacker()


@click.command()
@click.option("-t", "predict_target", flag_value="test", default=True)
@click.option("-v", "predict_target", flag_value="crossval")
@click.option("--location", "-l", type=str, default=None)
def main(predict_target, location):
    logger = logging.getLogger(__name__)
    logger.info('#0: train models')

    blender, stacker = gen_blender_and_stacker()
    if location is None:
        location_list = LOCATIONS
    else:
        location_list = [location, ]

    for place in location_list:
        # retrieve train y
        y_true_as_train = pd.read_csv(stacker.gen_y_true_filepath(place),
                                      **KWARGS_READ_CSV)
        y_true_as_train.dropna(axis=0, inplace=True)

        logger.info('#1: get y_true @ {l} !'.format(l=place))

        # retrieve train X
        df_pred_as_train = pd.read_csv(
            stacker.path.join(stacker.PROCESSED_DATA_BASEPATH,
                              "predict_y.layer_0.{t}.{l}.tsv".format(t=predict_target, l=place)),
            **KWARGS_READ_CSV
        )
        df_pred_as_train = df_pred_as_train.loc[y_true_as_train.index, ~df_pred_as_train.isnull().any()]

        logger.info('#1: get y_pred as a train data @ {l} !'.format(l=place))

        #
        # bifurcation
        #
        if predict_target == "crossval":
            # try cross-validation
            pd.DataFrame(
                blender.cross_val_predict(df_pred_as_train.values, y_true_as_train.values),
                index=df_pred_as_train.index,
                columns=[blend_model_name, ]
            ).to_csv(
                blender.gen_serialize_filepath("predict",
                                               "{t}.{l}.tsv".format(t=predict_target, l=place)),
                **KWARGS_TO_CSV
            )

            logger.info('#2: estimate y_pred of train dataset like cross-validation @ {l} !'.format(l=place))

        elif predict_target == "test":
            # fit model with the whole samples
            blender.fit(df_pred_as_train.values, y_true_as_train.values)

            logger.info('#2: fit & serialized a model @ {l} !'.format(l=place))

            # retrieve test X
            df_pred_as_test = pd.read_csv(
                stacker.path.join(stacker.PROCESSED_DATA_BASEPATH,
                                  "predict_y.layer_0.{t}.{l}.tsv".format(t=predict_target, l=place)),
                **KWARGS_READ_CSV
            )

            logger.info('#3: get y_pred as a test data @ {l} !'.format(l=place))

            # predict
            pd.DataFrame(
                blender.predict(df_pred_as_test.values),
                index=df_pred_as_test.index,
                columns=[blend_model_name, ]
            ).to_csv(
                blender.gen_serialize_filepath("predict",
                                               "{t}.{l}.tsv".format(t=predict_target, l=place)),
                **KWARGS_TO_CSV
            )

            logger.info('#4: estimate & save y_pred of test dataset @ {l} !'.format(l=place))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
