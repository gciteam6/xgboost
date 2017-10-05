# Built-in modules
from os import path, pardir
import logging
# third party modules
import click
from dotenv import find_dotenv, load_dotenv
# Hand-made modules
from unzip import Unzipper
from concatenation import DataFrameHandler
from amedas import AmedasPoint


AMD_MASTER_FILEPATH = "data/raw/amd_master.tsv"
HALF_MESHGRID_SIZE = 0.4


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

    # # Unzipping
    # unzipper = Unzipper()
    # for unzip_filepath in unzipper.gen_unzip_filepath_list(input_dirpath):
    #     logger.info('unzipping {f}'.format(f=path.basename(unzip_filepath)))
    #     unzipper.unzip(unzip_filepath, unzipper.INTERIM_DATA_BASEPATH)

    # # data concatenating
    # dfh = DataFrameHandler()
    # tsv_path = path.join(dfh.INTERIM_DATA_BASEPATH, "sfc1/sfc_47421.tsv")
    # dfh.set_tsvdata_of_every_10min(tsv_path)
    # print(dfh.df_list[0].head())

    # amedas information
    amd = AmedasPoint(path.join(project_dir, AMD_MASTER_FILEPATH))
    aid_near_ukishima = amd.get_near_amedas_points(
        amd.LATLNGALT_UKISHIMA[0], amd.LATLNGALT_UKISHIMA[1], HALF_MESHGRID_SIZE
    ).index
    aid_near_ougizima = amd.get_near_amedas_points(
        amd.LATLNGALT_OUGIZIMA[0], amd.LATLNGALT_OUGIZIMA[1], HALF_MESHGRID_SIZE
    ).index
    aid_near_yonekurayama = amd.get_near_amedas_points(
        amd.LATLNGALT_YONEKURAYAMA[0], amd.LATLNGALT_YONEKURAYAMA[1], HALF_MESHGRID_SIZE
    ).index


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = path.join(path.dirname(__file__), pardir, pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
