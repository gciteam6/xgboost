# Built-in modules
import gc
from os import path, pardir
import sys
import logging

# not used in this stub but often useful for finding various files
PROJECT_ROOT_DIRPATH = path.join(path.dirname(__file__), pardir, pardir)
sys.path.append(PROJECT_ROOT_DIRPATH)

# third party modules
import click
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import KFold
# Hand-made modules
from src.data.dataset import DatasetHandler
from src.data.unzip import Unzipper
from src.data.amedas import AmedasHandler
from src.data.surface import SurfaceHandler
from src.data.forecast import ForecastHandler
from src.data.sola import SolarPhotovoltaicHandler

AMD_MASTER_FILENAME = "amd_master.tsv"
SFC_MASTER_FILENAME = "sfc_master.tsv"
TRAIN_KWH_FILENAME = "train_kwh.tsv"
TRAIN_SEGMENTATION_MAX_NUMBER = 8
TEST_SEGMENTATION_MAX_NUMBER = 3

OBJECTIVE_LABEL_NAMES = ["kwh", ]
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

    maker = DatasetHandler(columns_y=OBJECTIVE_LABEL_NAMES)

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

        logger.info('#2: get amedas observation points @ {l} !'.format(l=location))

        amd_filepath = amd.gen_filepath_list(retrieve_amd_point.index)
        df_amd = amd.retrieve_data(amd_filepath, retrieve_amd_point["name"].as_matrix())

        df_data, df_flags = maker.split_data_and_flags(df_amd)
        maker.to_blp_via_df(
            df_data,
            path.join(maker.INTERIM_DATA_BASEPATH, "amd_data"),
            "{l}.data#2".format(l=location)
        )

        logger.info('#2: save amedas data as a middle file @ {l} !'.format(l=location))

        maker.to_blp_via_df(
            df_flags,
            path.join(maker.INTERIM_DATA_BASEPATH, "amd_flags"),
            "{l}.data#2".format(l=location)
        )

        logger.info('#2: save amedas flags as a middle file @ {l} !'.format(l=location))
        del (retrieve_amd_point, df_amd, df_data, df_flags)

    logger.info('#2: end amedas data processing !')
    del amd
    gc.collect()

    #
    # surface weather information
    #
    sfc = SurfaceHandler(path.join(input_dirpath, SFC_MASTER_FILENAME))

    for location, latlng_base_coordinate in zip(LOCATIONS, LATLNG_BASE_COORDINATES):
        retrieve_sfc_point = sfc.get_near_observation_points(
            latlng_base_coordinate[0], latlng_base_coordinate[1], sfc_half_mashgrid_size
        )

        logger.info('#3: get surface weather observation points @ {l} !'.format(l=location))

        sfc_filepath = sfc.gen_filepath_list(retrieve_sfc_point.index)
        df_sfc = sfc.retrieve_data(sfc_filepath, retrieve_sfc_point["name"].as_matrix())

        df_data, df_flags = maker.split_data_and_flags(df_sfc)
        maker.to_blp_via_df(
            df_data,
            path.join(maker.INTERIM_DATA_BASEPATH, "sfc_data"),
            "{l}.data#3".format(l=location)
        )

        logger.info('#3: save surface weather data as a middle file @ {l} !'.format(l=location))

        maker.to_blp_via_df(
            df_flags,
            path.join(maker.INTERIM_DATA_BASEPATH, "sfc_flags"),
            "{l}.data#3".format(l=location)
        )

        logger.info('#3: save surface weather flags as a middle file @ {l} !'.format(l=location))
        del (retrieve_sfc_point, df_sfc, df_data, df_flags)

    logger.info('#3: end surface weather data processing !')
    del sfc
    gc.collect()

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
                sr_expand_whole_day_data.index,
                whole_day_data_name
            ] = sr_expand_whole_day_data

            logger.info('#4: expand "{n}" @ {l}'.format(n=whole_day_data_name, l=location))
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

            logger.info('#4: expand "{n}" @ {l}'.format(n=time_ranged_data_name, l=location))
            del sr_expand_time_ranged_data

        df_forecast_expanded.drop(time_ranged_data_name_list, axis=1, inplace=True)
        maker.to_blp_via_df(
            df_forecast_expanded,
            path.join(maker.INTERIM_DATA_BASEPATH, "forecast_data"),
            "{l}.data#4".format(l=location)
        )

        logger.info('#4: save forecast data as a middle file @ {l} !'.format(l=location))
        del (
            df_forecast,
            df_forecast_expanded,
            whole_day_data_name_list,
            time_ranged_data_name_list
        )

    logger.info('#4: end forecast data processing !')
    del forecast
    gc.collect()

    #
    # train_kwh information
    #
    sola = SolarPhotovoltaicHandler()
    df_train_kwh = sola.read_tsv(path.join(input_dirpath, TRAIN_KWH_FILENAME))

    for location in LOCATIONS:
        col_label = sola.SOLA_LOCATION_LABEL_NAMES[location]
        df_sola = df_train_kwh.loc[:, col_label].to_frame(name=maker.columns_y[0])
        maker.to_blp_via_df(
            df_sola,
            path.join(maker.INTERIM_DATA_BASEPATH, "sola_data"),
            "{l}.data#5".format(l=location)
        )

        logger.info('#5: save solar data as a middle file @ {l}!'.format(l=location))
        del df_sola

    logger.info('#5: end solar data processing !')
    del (sola, df_train_kwh)
    gc.collect()

    #
    # gether data
    #
    kf_train = KFold(n_splits=TRAIN_SEGMENTATION_MAX_NUMBER)
    kf_test = KFold(n_splits=TEST_SEGMENTATION_MAX_NUMBER)

    for location in LOCATIONS:
        filepath_prefix_suffix_nested_list = \
            maker.gen_filepath_prefix_suffix_nested_list(location)
        df_train_segment, _ = maker.separate_train_test(
            maker.retrieve_data(filepath_prefix_suffix_nested_list)
        )

        logger.info('#6: load train data segment to the memory @ {l} !'.format(l=location))

        for n_iter, (_, extract_index) in enumerate(kf_train.split(df_train_segment)):
            maker.to_blp_via_df(
                df_train_segment.iloc[extract_index, :],
                path.join(maker.PROCESSED_DATA_BASEPATH,
                          "dataset.amd_sfc_forecast_kwh.train{i}".format(i=n_iter)),
                "{l}.blp".format(l=location)
                )

            logger.info('#6: save separated data of train{i} @ {l} !'.format(i=n_iter, l=location))

        logger.info('#6: end processing of train data segment @ {l} !'.format(l=location))
        del df_train_segment
        gc.collect()

        _, df_test_segment = maker.separate_train_test(
            maker.retrieve_data(filepath_prefix_suffix_nested_list)
        )

        logger.info('#6: load test data segment to the memory @ {l} !'.format(l=location))

        for n_iter, (_, extract_index) in enumerate(kf_test.split(df_test_segment)):
            maker.to_blp_via_df(
                df_test_segment.iloc[extract_index, :],
                path.join(maker.PROCESSED_DATA_BASEPATH,
                          "dataset.amd_sfc_forecast_kwh.test{i}".format(i=n_iter)),
                "{l}.blp".format(l=location)
                )

            logger.info('#6: save separated data of test{i} @ {l} !'.format(i=n_iter, l=location))

        logger.info('#6: end processing of test data segment @ {l} !'.format(l=location))
        del df_test_segment
        gc.collect()

    logger.info('#6: end src/data/make_dataset.py !')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
