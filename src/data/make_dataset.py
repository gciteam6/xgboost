# Built-in modules
import os
from os import path
import logging
# third party modules
import click
from dotenv import find_dotenv, load_dotenv
# Hand-made modules
from unzip import Unzipper
from concatenation import DataFrameHandler
from latlng import AmedasPoint

@click.command()
@click.argument("input_dirpath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
# @click.argument("--output_filepath", "-o", type=click.Path())
def main(input_dirpath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Unzipping
    unzipper = Unzipper()
    for unzip_filepath in unzipper.gen_unzip_filepath_list(input_dirpath):
        logger.info('unzipping {f}'.format(f=path.basename(unzip_filepath)))
        unzipper.unzip(unzip_filepath, unzipper.INTERIM_DATA_BASEPATH)

    # data concatenating
    dfh = DataFrameHandler()
    tsv_path = path.join(dfh.INTERIM_DATA_BASEPATH, "sfc1/sfc_47421.tsv")
    dfh.set_tsvdata_of_every_10min(tsv_path)
    print(dfh.df_list[0].head())

    #
    amd_master_filepath = path.join(project_dir, "data/raw/amd_master.tsv")
    amd = AmedasPoint(amd_master_filepath)
    print(amd.df_amedas_point.head())

    print(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = path.join(path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
