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
# Hand-made modules
from src.models.split import DatasetSplitHandler


TRAIN_FILEPATH_PREFIX = path.join(
    PROJECT_ROOT_DIRPATH, "data/interim/dataset.train_X_y"
)
LOCATIONS = (
    "ukishima",
    "ougishima",
    "yonekurayama"
)


@click.command()
@click.option("--location", "-l", type=str, default=None)
@click.option("--n_splits", "-n", type=int, default=5)
def main(location, n_splits):
    logger = logging.getLogger(__name__)
    logger.info('#0: separating X and y from the train data set')

    #
    # split train X and y
    #
    splitter = DatasetSplitHandler()

    if location is None:
        location_list = LOCATIONS
    else:
        location_list = [location, ]

    for place in location_list:
        train_filepath_prefix = path.join(
            PROJECT_ROOT_DIRPATH, "data/processed/dataset.train_X_y"
        )
        splitter.separate_and_save_train_y(train_filepath_prefix, place)

        logger.info('#1: end separating & saving train y @ {l}'.format(l=place))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
