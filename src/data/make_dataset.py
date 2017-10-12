# Built-in modules
from os import path, pardir
import sys
import logging

# not used in this stub but often useful for finding various files
PROJECT_ROOT_DIRPATH = path.join(path.dirname(__file__), pardir, pardir)
sys.path.append(PROJECT_ROOT_DIRPATH)

# third party modules
import click
from dotenv import find_dotenv, load_dotenv
# Hand-made modules
from src.data.unzip import Unzipper
from src.data.amedas import AmedasHandler
from src.data.surface import SurfaceHandler
from src.data.forecast import ForecastHandler
from src.data.sola import SolarPhotovoltaicHandler
from src.data.concatenation import DatasetCollector

AMD_MASTER_FILENAME = "amd_master.tsv"
SFC_MASTER_FILENAME = "sfc_master.tsv"
TRAIN_KWH_FILENAME = "train_kwh.tsv"
OBJECTIVE_COLUMN_NAME = "kwh"

LATLNGALT_UKISHIMA = (35.517558, 139.786920, 4.7)
LATLNGALT_OUGISHIMA = (35.488680, 139.727451, 4.8)
LATLNGALT_YONEKURAYAMA = (35.583302, 138.573118, 366.9)

LOCATIONS = (
    "ukishima",
    "ougishima",
    "yonekurayama",
)
LATLNG_BASE_COORDINATES = (
    LATLNGALT_UKISHIMA,
    LATLNGALT_OUGISHIMA,
    LATLNGALT_YONEKURAYAMA,
)


@click.command()
@click.argument("input_dirpath", type=click.Path(exists=True))
@click.option("--amd_half_mashgrid_size", "-a", type=float, default=0.2)
@click.option("--sfc_half_mashgrid_size", "-s", type=float, default=0.4)
@click.option("--is_unzip", "-z", type=bool, default=True)
def main(input_dirpath,
         amd_half_mashgrid_size,
         sfc_half_mashgrid_size,
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

        logger.info('#1: end unzip !')
        del unzipper
    else:
        logger.info('#1: skipped unzipping !')

    #
    # amedas information
    #
    amd = AmedasHandler(path.join(input_dirpath, AMD_MASTER_FILENAME))

    for location, latlng_base_coordinate in zip(LOCATIONS, LATLNG_BASE_COORDINATES):
        retrieve_amd_point = amd.get_near_observation_points(
            latlng_base_coordinate[0], latlng_base_coordinate[1], amd_half_mashgrid_size
        )

        logger.info('#2: get amedas observation points in {l} !'.format(l=location))

        amd_filepath = amd.gen_filepath_list(retrieve_amd_point.index)
        df_amd = amd.retrieve_data(amd_filepath, retrieve_amd_point["name"].as_matrix())
        amd.to_tsv(df_amd, path.join(amd.INTERIM_DATA_BASEPATH, "amd_data_near.{l}.tsv".format(l=location)))

        logger.info('#2: gather amedas data and save as a middle file in {l} !'.format(l=location))
        del (df_amd, retrieve_amd_point)

    logger.info('#2: end amedas data processing !')
    del amd

    #
    # surface weather information
    #
    sfc = SurfaceHandler(path.join(input_dirpath, SFC_MASTER_FILENAME))

    for location, latlng_base_coordinate in zip(LOCATIONS, LATLNG_BASE_COORDINATES):
        retrieve_sfc_point = sfc.get_near_observation_points(
            latlng_base_coordinate[0], latlng_base_coordinate[1], sfc_half_mashgrid_size
        )

        logger.info('#3: get surface weather observation points in {l} !'.format(l=location))

        sfc_filepath = sfc.gen_filepath_list(retrieve_sfc_point.index)
        df_sfc = sfc.retrieve_data(sfc_filepath, retrieve_sfc_point["name"].as_matrix())
        sfc.to_tsv(df_sfc, path.join(sfc.INTERIM_DATA_BASEPATH, "sfc_data_near.{l}.tsv".format(l=location)))

        logger.info('#3: gather surface weather data and save as a middle file in {l} !'.format(l=location))
        del (df_sfc, retrieve_sfc_point)

    logger.info('#3: end surface weather data processing !')
    del sfc

    #
    # forecast information
    #
    forecast = ForecastHandler()

    for location in LOCATIONS:
        forecast_filepath = forecast.gen_filepath(location)
        df_forecast = forecast.read_tsv(forecast_filepath)
        df_forecast_expanded = forecast.add_datetime_ticks(df_forecast)

        whole_day_data_name_list = \
            forecast.get_whole_day_data_columns(df_forecast.columns)

        for whole_day_data_name in whole_day_data_name_list:
            sr_expand_whole_day_data = \
                forecast.expand_whole_day_data(df_forecast[whole_day_data_name])
            df_forecast_expanded.loc[
                sr_expand_whole_day_data.index, whole_day_data_name
            ] = sr_expand_whole_day_data

            logger.info('#4: expand "{n}" in forecast'.format(n=whole_day_data_name))
            del sr_expand_whole_day_data

        time_ranged_data_name_list = \
            forecast.get_time_ranged_data_columns(df_forecast.columns)

        for time_ranged_data_name in time_ranged_data_name_list:
            sr_expand_time_ranged_data = \
                forecast.expand_time_ranged_data(df_forecast[time_ranged_data_name])
            df_forecast_expanded.loc[
                sr_expand_time_ranged_data.index,
                forecast.extract_attribute_from_time_ranged_column_name(time_ranged_data_name)
            ] = sr_expand_time_ranged_data

            logger.info('#4: expand "{n}" in {l}'.format(n=time_ranged_data_name, l=location))
            del sr_expand_time_ranged_data

        df_forecast_expanded.drop(time_ranged_data_name_list, axis=1, inplace=True)
        forecast.to_tsv(
            df_forecast_expanded,
            path.join(forecast.INTERIM_DATA_BASEPATH, "forecast_data.{l}.tsv".format(l=location))
        )

        logger.info('#4: gather forecast data and save as a middle file in {l} !'.format(l=location))
        del (
            df_forecast,
            df_forecast_expanded,
            whole_day_data_name_list,
            time_ranged_data_name_list
        )

    logger.info('#4: end forecast data processing !')
    del forecast

    #
    # train_kwh information
    #
    sola = SolarPhotovoltaicHandler()
    df_train_kwh = sola.read_tsv(path.join(input_dirpath, TRAIN_KWH_FILENAME))

    for col_label, location in zip(
            [sola.LABEL_SOLA_UKISHIMA, sola.LABEL_SOLA_OUGISHIMA, sola.LABEL_SOLA_YONEKURAYAMA],
            LOCATIONS):
        df_sola = df_train_kwh.loc[:, col_label].to_frame(name=OBJECTIVE_COLUMN_NAME)
        sola.to_tsv(df_sola, path.join(sola.INTERIM_DATA_BASEPATH, "sola_data.{l}.tsv".format(l=location)))

        logger.info('#5: get solar data and save as a middle file in {l}!'.format(l=location))
        del df_sola

    logger.info('#5: end solar data processing !')
    del (sola, df_train_kwh)

    #
    # gether data
    #
    collector = DatasetCollector()

    for location in LOCATIONS:
        location_filepath = collector.gen_filepath_list(location)
        df_train_for_each_location = collector.retrieve_data(location_filepath)
        collector.to_tsv(
            df_train_for_each_location,
            path.join(
                collector.PROCESSED_DATA_BASEPATH,
                "dataset.amd_sfc_forecast_kwh.{l}.tsv".format(l=location)
            )
        )

        logger.info('#6: generate and save the dataset as a processed file in {l} !'.format(l=location))
        del df_train_for_each_location


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
