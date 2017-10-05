# Built-in modules
from os import path, pardir
import logging
# third party modules
import click
from dotenv import find_dotenv, load_dotenv
# Hand-made modules
from unzip import Unzipper
from concatenation import DataFrameHandler
from amedas import AmedasData, AmedasPoint


AMD_MASTER_FILEPATH = "data/raw/amd_master.tsv"
HALF_MESHGRID_SIZE = 0.2


@click.command()
@click.argument("input_dirpath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
# @click.argument("--half_mashgrid_size", "-m", type=float, default=0.2)
def main(input_dirpath, output_filepath, half_mashgrid_size):
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
    amedas_point = AmedasPoint(path.join(project_dir, AMD_MASTER_FILEPATH))
    amedas_data = AmedasData()
    amd_point_near_ukishima = amedas_point.get_near_amedas_points(
        amedas_point.LATLNGALT_UKISHIMA[0], amedas_point.LATLNGALT_UKISHIMA[1], half_mashgrid_size
    )
    # amd_point_near_ougizima = amedas_point.get_near_amedas_points(
    #     amedas_point.LATLNGALT_OUGIZIMA[0], amedas_point.LATLNGALT_OUGIZIMA[1], half_mashgrid_size
    # )
    # amd_point_near_yonekurayama = amd.get_near_amedas_points(
    #     amedas_point.LATLNGALT_YONEKURAYAMA[0], amedas_point.LATLNGALT_YONEKURAYAMA[1], half_mashgrid_size
    # )

    amd_filepath = amedas_data.gen_filepath_list(amd_point_near_ukishima.index)
    df_amd = amedas_data.retrive_amedas_data(amd_filepath, amd_point_near_ukishima["name"].as_matrix())
    amedas_data.to_tsv(df_amd, path.join(amedas_data.INTERIM_DATA_BASEPATH, "amd_data_near_ukishima.tsv"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = path.join(path.dirname(__file__), pardir, pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
