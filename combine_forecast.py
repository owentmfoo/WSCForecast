import logging

import awswrangler as wr
import pandas as pd
from S5 import Tecplot as tp
from S5.Weather import convert_wind
import numpy as np
import boto3
from datetime import datetime, timezone, timedelta
from pvlib import solarposition
import warnings

logging.getLogger().setLevel(logging.INFO)

# s3 = boto3.client('s3')
# TODO: edit these before uploading to aws
race_start = datetime(2023, 8, 17, tzinfo=timezone.utc)
race_end = datetime(2023, 8, 23, tzinfo=timezone.utc)
startime = race_start - timedelta(1)
output_file = 'Weather-DEV1.dat'

# tomorrow = wr.s3.read_parquet("s3://solcastresults/tomorrow/", dataset=True,
#                               last_modified_begin=startime)
# solcast = wr.s3.read_parquet("s3://solcastresults/solcast/", dataset=True,
#                              last_modified_begin=startime)

tomorrow = pd.read_parquet('tomorrow.parquet')
solcast = pd.read_parquet('solcast.parquet')

tomorrow.rename(
    columns={'dni': 'DirectSun (W/m2)', 'dhi': 'DiffuseSun (W/m2)',
             'temperature': 'AirTemp (degC)',
             'pressureSurfaceLevel': 'AirPress (hPa)',
             'windSpeed': '10m WindVel (m/s)',
             'windDirection': 'WindDir (deg)',
             'Azimuth': 'SunAzimuth (deg)',
             'latitude': 'Latitude',
             'longitude': 'Longitude'}, inplace=True)
tomorrow.loc[:, 'WindVel (m/s)'] = convert_wind(tomorrow.loc[:, '10m WindVel (m/s)'])
tomorrow.loc[:, 'AirPress (Pa)'] = tomorrow.loc[:, 'AirPress (hPa)'] * 100

solcast.rename(
    columns={'dni': 'DirectSun (W/m2)', 'dhi': 'DiffuseSun (W/m2)',
             'temperature': 'AirTemp (degC)',
             'pressureSurfaceLevel': 'AirPress (hPa)',
             'windSpeed': '10m WindVel (m/s)',
             'windDirection': 'WindDir (deg)',
             'Azimuth': 'SunAzimuth (deg)',
             'latitude': 'Latitude',
             'longitude': 'Longitude'}, inplace=True)

road = tp.TecplotData("E:\WSCForecast\RoadFile-LatLon-2021.dat")
road.data.set_index('Distance (km)', inplace=True)


def extract_data(forecast_source: pd.DataFrame) -> pd.DataFrame:
    spots = forecast_source[["Latitude", "Longitude"]].drop_duplicates()
    for i, spot in spots.iterrows():
        distance = ((road.data.Longitude - spot.Longitude) ** 2 + (
                road.data.Latitude - spot.Latitude) ** 2)

        # associate the spot to the distance along the route that it is closest
        dist_along_route = road.data.index[distance.argmin()]

        # associate the spot to a distance along the route if it within 10m
        if distance.min() < 0.01:
            forecast_source.loc[
                forecast_source.Latitude == spot.Latitude, "Distance (km)"] = dist_along_route
            logging.debug(
                "Associated spot {}, {} with {} km".format(spot.Latitude,
                                                           spot.Longitude,
                                                           dist_along_route))
        else:
            forecast_source.loc[
                forecast_source.Latitude == spot.Latitude, "Distance (km)"] = np.NAN
            logging.info(
                'Dropping spot {}, {}'.format(spot.Latitude, spot.Longitude))

    forecast_source.dropna(subset='Distance (km)', inplace=True)
    # keep the latest forecast value
    forecast_source = forecast_source.sort_values(
        by=["period_end", "prediction_date"]).drop_duplicates(
        subset=['period_end', 'Distance (km)'], keep='last')
    forecast_source.set_index('period_end', inplace=True)
    return forecast_source.copy()


def calc_sunpos(forecast):
    solpos = solarposition.get_solarposition(forecast.index,
                                             forecast.Latitude,
                                             forecast.Longitude)
    solpos.loc[:, 'azimuth'] = solpos['azimuth'].apply(
        lambda x: x if x < 180 else x - 360)
    forecast.loc[:, "SunAzimuth (deg)"] = solpos['azimuth']
    forecast.loc[:, "SunElevation (deg)"] = solpos['apparent_elevation']
    return forecast


solcast = extract_data(
    solcast.loc[:, ['DirectSun (W/m2)', 'DiffuseSun (W/m2)', 'Longitude', 'Latitude', 'period_end',
                    'prediction_date']])
solcast.index = solcast.index.tz_localize('UTC')


tomorrow = extract_data(tomorrow.loc[:, ['AirPress (Pa)', 'AirTemp (degC)',
                                         'WindDir (deg)', 'WindVel (m/s)',
                                         'Longitude', 'Latitude', 'period_end',
                                         'prediction_date']])



# drop prediction time col before joining
solcast_prediction_time = solcast.prediction_date.max()
solcast.drop(columns=["prediction_date"], inplace=True)
tomorrow_prediction_time = tomorrow.prediction_date.max()
tomorrow.drop(columns=["prediction_date"], inplace=True)

forecast = pd.merge(tomorrow, solcast.astype(np.float64).drop(
    columns=['Longitude', 'Latitude']), how='outer',
                    on=['period_end', 'Distance (km)'])
forecast = forecast.reset_index().set_index(
    ['Distance (km)', 'period_end']).sort_values(
    by=['Distance (km)', 'period_end'])

weather = tp.SSWeather()
# weather.data = pd.DataFrame(columns=forecast.columns)
# split it up per location and interpolate to fill gaps.
for dist in forecast.index.levels[0].unique():
    spot_forecast = forecast.loc[dist].interpolate()
    spot_forecast.loc[:, "Distance (km)"] = dist
    weather.data = pd.concat(
        [weather.data, spot_forecast.loc[race_start:race_end]])

# TODO: what if there are more spots from tomorrow than from solcast?
#  need spatial interpolation as well

# Pad to 0km and 3030km
dist = forecast.index.levels[0].unique().min()
spot_forecast = forecast.loc[dist].interpolate()
spot_forecast.loc[:, "Distance (km)"] = 0
weather.data = pd.concat([weather.data, spot_forecast.loc[race_start:race_end]])

dist = forecast.index.levels[0].unique().max()
spot_forecast = forecast.loc[dist].interpolate()
spot_forecast.loc[:, "Distance (km)"] = 3030
weather.data = pd.concat([weather.data, spot_forecast.loc[race_start:race_end]])


missing = weather.data.loc[weather.data.isna().sum(axis=1) > 0]
if missing.index.shape != (0,):
    warnings.warn("missing values present")  # TODO: add more info for debugging

weather.data = calc_sunpos(weather.data)

# weather.data.rename(
#     columns={'dni': 'DirectSun (W/m2)', 'dhi': 'DiffuseSun (W/m2)',
#              'temperature': 'AirTemp (degC)',
#              'pressureSurfaceLevel': 'AirPress (hPa)',
#              'windSpeed': '10m WindVel (m/s)',
#              'windDirection': 'WindDir (deg)',
#              'Azimuth': 'SunAzimuth (deg)',
#              'latitude': 'Latitude',
#              'longitude': 'Longitude'}, inplace=True)
#
# weather.data.loc[:, 'WindVel (m/s)'] = convert_wind(weather.data.loc[:, '10m WindVel (m/s)'])
# weather.data.loc[:, 'AirPress (Pa)'] = weather.data.loc[:, 'AirPress (hPa)'] * 100
# TODO: check azimuth

weather.data = weather.data[['Distance (km)', 'DirectSun (W/m2)', 'DiffuseSun (W/m2)',
         'SunAzimuth (deg)',
         'SunElevation (deg)', 'AirTemp (degC)', 'AirPress (Pa)',
         'WindVel (m/s)',
         'WindDir (deg)']].copy()

weather.data.sort_values(by=['Distance (km)', 'period_end'], inplace=True)
if race_start.tzinfo != race_end.tzinfo:
    warnings.warn(
        'Starting and ending time zone mismatch, using starting timezone as output timezone.')
output_timezone = race_start.tzinfo
weather.data = weather.data.tz_convert(output_timezone)
weather.add_day_time_cols()

weather.data.reset_index(inplace=True)
weather.data.drop(columns='period_end', inplace=True)
weather.zone.nj = weather.data.loc[:, 'Distance (km)'].nunique()
weather.zone.ni = weather.data.iloc[:, 0].count() / weather.zone.nj

weather.data = weather.data[
    ['Day', 'Time (HHMM)', 'Distance (km)', 'DirectSun (W/m2)',
     'DiffuseSun (W/m2)', 'SunAzimuth (deg)',
     'SunElevation (deg)', 'AirTemp (degC)', 'AirPress (Pa)', 'WindVel (m/s)',
     'WindDir (deg)']]

weather.check_rectangular()
# add tecplot title inc forecast dates

weather.title = f'WSC forecast starting at {race_start}, solcast forecast from {solcast_prediction_time}, tomorrow forecast from {tomorrow_prediction_time}.'

weather.write_tecplot(output_file)

