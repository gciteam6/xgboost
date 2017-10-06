# Built-in modules
from os import path, pardir
import logging
# third party modules
import click
from dotenv import find_dotenv, load_dotenv
# Hand-made modules
from unzip import Unzipper
from amedas import AmedasHandler
from surface import SurfaceHandler
from sola import SolarPhotovoltaicHandler


AMD_MASTER_FILEPATH = "data/raw/amd_master.tsv"
SFC_MASTER_FILEPATH = "data/raw/sfc_master.tsv"
TRAIN_KWH_FILEPATH = "data/raw/train_kwh.tsv"
LOCATION = [
    "ukishima",
    "ougishima",
    "yonekurayama"
]


@click.command()
@click.argument("input_dirpath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--amd_half_mashgrid_size", "-a", type=float, default=0.2)
@click.option("--scf_half_mashgrid_size", "-s", type=float, default=0.4)
@click.option("--is_unzip", "-z", type=bool, default=False)
def main(input_dirpath, output_filepath,
         amd_half_mashgrid_size, scf_half_mashgrid_size,
         is_unzip):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('#0: making final data set from raw data')

    #
    # Unzipping
    #
    if is_unzip:
        unzipper = Unzipper()
        for unzip_filepath in unzipper.gen_unzip_filepath_list(input_dirpath):
            unzipper.unzip(unzip_filepath, unzipper.INTERIM_DATA_BASEPATH)
            logger.info('#1: unzipped {f} !'.format(f=path.basename(unzip_filepath)))
    else:
        logger.info('#1: skipped unzipping !')

    #
    # amedas information
    #
    amd = AmedasHandler(path.join(project_dir, AMD_MASTER_FILEPATH))

    amd_point_near_ukishima = amd.get_near_observation_points(
        amd.LATLNGALT_UKISHIMA[0], amd.LATLNGALT_UKISHIMA[1], amd_half_mashgrid_size
    )
    amd_point_near_ougishima = amd.get_near_observation_points(
        amd.LATLNGALT_OUGISHIMA[0], amd.LATLNGALT_OUGISHIMA[1], amd_half_mashgrid_size
    )
    amd_point_near_yonekurayama = amd.get_near_observation_points(
        amd.LATLNGALT_YONEKURAYAMA[0], amd.LATLNGALT_YONEKURAYAMA[1], amd_half_mashgrid_size
    )
    logger.info('#2: get amedas observation points !')

    for amd_point, location in zip(
            [amd_point_near_ukishima, amd_point_near_ougishima, amd_point_near_yonekurayama],
            LOCATION):
        amd_filepath = amd.gen_filepath_list(amd_point.index)
        df_amd = amd.retrieve_data(amd_filepath, amd_point["name"].as_matrix())
        amd.to_tsv(df_amd, path.join(amd.INTERIM_DATA_BASEPATH, "amd_data_near_{l}.tsv".format(l=location)))
        logger.info('#3: gather amedas data and save as a middle file in {l} !'.format(l=location))

    #
    # surface weather information
    #
    sfc = SurfaceHandler(path.join(project_dir, SFC_MASTER_FILEPATH))

    sfc_point_near_ukishima = sfc.get_near_observation_points(
        sfc.LATLNGALT_UKISHIMA[0], sfc.LATLNGALT_UKISHIMA[1], scf_half_mashgrid_size
    )
    sfc_point_near_ougishima = sfc.get_near_observation_points(
        sfc.LATLNGALT_OUGISHIMA[0], sfc.LATLNGALT_OUGISHIMA[1], scf_half_mashgrid_size
    )
    sfc_point_near_yonekurayama = sfc.get_near_observation_points(
        sfc.LATLNGALT_YONEKURAYAMA[0], sfc.LATLNGALT_YONEKURAYAMA[1], scf_half_mashgrid_size
    )
    logger.info('#4: get surface weather observation points !')

    for sfc_point, location in zip(
            [sfc_point_near_ukishima, sfc_point_near_ougishima, sfc_point_near_yonekurayama],
            LOCATION):
        sfc_filepath = sfc.gen_filepath_list(sfc_point.index)
        df_sfc = sfc.retrieve_data(sfc_filepath, sfc_point["name"].as_matrix())
        sfc.to_tsv(df_sfc, path.join(sfc.INTERIM_DATA_BASEPATH, "sfc_data_near_{l}.tsv".format(l=location)))
        logger.info('#5: gather surface weather data and save as a middle file in {l} !'.format(l=location))

    #
    # train_kwh information
    #
    sola = SolarPhotovoltaicHandler()
    df_train_kwh = sola.read_tsv(path.join(project_dir, TRAIN_KWH_FILEPATH))

    for col_label, location in zip(
            [sola.LABEL_SOLA_UKISHIMA, sola.LABEL_SOLA_OUGISHIMA, sola.LABEL_SOLA_YONEKURAYAMA],
            LOCATION):
        df_sola = df_train_kwh.loc[:, col_label]
        sola.to_tsv(df_sola, path.join(sfc.INTERIM_DATA_BASEPATH, "sola_data_{l}.tsv".format(l=location)))

    logger.info('#6: get solar data and save as a middle file !')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = path.join(path.dirname(__file__), pardir, pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
