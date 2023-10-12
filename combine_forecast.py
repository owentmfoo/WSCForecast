"""Combines the forecast gathered from solcast and tomorrow.io

Combines the forecast gathered from solcast and tomorrow.io and publishes it to
S3.

- obtains forecasts stored on S3 by get_solcast_forecast and
get_tomorrow_forecast.
- outer merge between the two dataset and linearly interpolate to fill gaps
- convert wind from 10m to 1m
- publishes the weather file to S3 and set it to public, "Weather-Latest.dat"
gets overwritten and it also saves a timestamped copy.
- title of the weather file contains timestamps of when the forecast was
made and when the weather file is created.
"""
import logging
from datetime import datetime, timezone, timedelta

import awswrangler as wr
import pandas as pd
from S5 import Tecplot as tp
from S5.Weather import convert_wind
import numpy as np
import boto3
from pvlib import solarposition
import pytz
from scipy import interpolate
from utils import get_locations

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

s5_logger = logging.getLogger("S5")
s5_logger.setLevel(logging.DEBUG)
s5_logger.addHandler(handler)


def extract_data(forecast_source: pd.DataFrame, road: pd.DataFrame) -> pd.DataFrame:
    """Maps forecast locations to distance along the route and keep the latest
    forecast.

    Maps the lat,long from the forecast to distance along the route by taking
    the argmin of the euclidian distance. If the spot is more than 0.1 km away
    from the route, the spot will be dropped.
    Only the latest forecast of each spot will be kept.

    Args:
        forecast_source: DataFrame containing columns "period_end",
        "prediction_date","Latitude", and "Longitude"
        road: DataFame that maps "Latitude" and "Longitude" to "Distance (km)".

    Returns:
        Trimmed DataFrame with column "Distance (km)" added.
    Raises:
        KeyError:If columns are missing from the input DataFrames.

    """
    if ("Distance (km)" not in road.columns) and (road.index.name != "Distance (km)"):
        raise KeyError("road missing Distance (km)")
    if road.index.name != "Distance (km)":
        road.set_index("Distance (km)", inplace=True)

    spots = forecast_source[["Latitude", "Longitude"]].drop_duplicates()
    for _, spot in spots.iterrows():
        distance = np.sqrt(
            (road.Longitude - spot.Longitude) ** 2
            + (road.Latitude - spot.Latitude) ** 2
        )

        # get the point the is closest along the route
        dist_along_route = road.index[distance.argmin()]

        # associate the spot to a distance along the route if it within 100m
        if distance.min() < 0.1:
            forecast_source.loc[
                forecast_source.Latitude == spot.Latitude, "Distance (km)"
            ] = dist_along_route
            logging.debug(
                "Associated spot %f, %f with %f km.",
                spot.Latitude,
                spot.Longitude,
                dist_along_route,
            )
        else:
            forecast_source.loc[
                forecast_source.Latitude == spot.Latitude, "Distance (km)"
            ] = np.NAN
            logging.info(
                "Dropping spot %f, %f, spot is %f km away from the route.",
                spot.Latitude,
                spot.Longitude,
                distance.min(),
            )

    forecast_source.dropna(subset="Distance (km)", inplace=True)
    # keep the latest forecast value
    forecast_source = forecast_source.sort_values(
        by=["period_end", "prediction_date"]
    ).drop_duplicates(subset=["period_end", "Distance (km)"], keep="last")
    forecast_source.set_index("period_end", inplace=True)
    return forecast_source.copy()


def calc_sunpos(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the azimuth and elevation for the DataFrame.

    This a wrapper function that takes the DateTimeIndex, Latitude and Longitude
    columns from the DataFrame and pass it into the get_solarposition function
    in pvlib. The calculated Azimuth and Elevation will be added back into the
    original DataFrame as columns "SunAzimuth (deg)" and "SunElevation (deg)".

    Args:
        df: DataFrame with a DateTimeIndex, and columns named "Latitude" and
    "Longitude".

    Returns:
        DataFrame with columns "SunAzimuth (deg)" and "SunElevation (deg)"added
        to it.

    Raises:
        AttributeError: If the DataFrame index is not pd.DateTimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AttributeError("DataFrame index must DateTimeIndex.")
    sun_pos = solarposition.get_solarposition(df.index, df.Latitude, df.Longitude)
    sun_pos.loc[:, "azimuth"] = sun_pos["azimuth"].apply(
        lambda x: x if x < 180 else x - 360
    )
    df.loc[:, "SunAzimuth (deg)"] = sun_pos["azimuth"]
    df.loc[:, "SunElevation (deg)"] = sun_pos["apparent_elevation"]
    # TODO: add the other outputs as well to future proof this?
    # TODO: check azimuth of the output files, sign conventions in SolarSim?
    return df


def combine_forecast(
    solcast: pd.DataFrame,
    tomorrow: pd.DataFrame,
    road: pd.DataFrame,
    race_start: datetime,
    race_end: datetime,
) -> tp.SSWeather:
    """Combines forecast from solcast and tomorrow.io

    Args:
        solcast: forecast from Solcast
        tomorrow: forecast from tomorrow.io
        road: road DataFrame to map lat lon to dist
        race_start: race start date with tz localised
        race_end: race end date with tz localised

    Returns:
        Tecplot weather file object.
    """
    tomorrow.rename(
        columns={
            "dni": "DirectSun (W/m2)",
            "dhi": "DiffuseSun (W/m2)",
            "temperature": "AirTemp (degC)",
            "pressureSurfaceLevel": "AirPress (hPa)",
            "windSpeed": "10m WindVel (m/s)",
            "windDirection": "WindDir (deg)",
            "Azimuth": "SunAzimuth (deg)",
            "latitude": "Latitude",
            "longitude": "Longitude",
        },
        inplace=True,
    )
    tomorrow.loc[:, "WindVel (m/s)"] = convert_wind(
        tomorrow.loc[:, "10m WindVel (m/s)"]
    )
    tomorrow.loc[:, "AirPress (Pa)"] = tomorrow.loc[:, "AirPress (hPa)"] * 100

    solcast.rename(
        columns={
            "dni": "DirectSun (W/m2)",
            "dhi": "DiffuseSun (W/m2)",
            "air_temp": "AirTemp (degC)",
            "pressureSurfaceLevel": "AirPress (hPa)",
            "windSpeed": "10m WindVel (m/s)",
            "windDirection": "WindDir (deg)",
            "Azimuth": "SunAzimuth (deg)",
            "latitude": "Latitude",
            "longitude": "Longitude",
        },
        inplace=True,
    )
    solcast.loc[:, "WindVel (m/s)"] = convert_wind(solcast.loc[:, "10m WindVel (m/s)"])
    solcast.loc[:, "AirPress (Pa)"] = solcast.loc[:, "AirPress (hPa)"] * 100

    solcast = extract_data(
        solcast.loc[
            :,
            [
                "DirectSun (W/m2)",
                "DiffuseSun (W/m2)",
                "Longitude",
                "Latitude",
                "period_end",
                "prediction_date",
                "AirTemp (degC)",
                "AirPress (Pa)",
                "WindVel (m/s)",
                "WindDir (deg)",
                "dni10",
                "dni90",
                "dhi10",
                "dhi90",
            ],
        ],
        road,
    )
    if solcast.index.tz is None:
        solcast.index = solcast.index.tz_localize("UTC")

    # shift the solcast timestamps forward bu 15 min so the values corresponds
    #  the centre of the period.
    # TODO: check period and adjust accordingly
    solcast = solcast.shift(-15, "min")

    tomorrow = extract_data(
        tomorrow.loc[
            :,
            [
                "AirPress (Pa)",
                "AirTemp (degC)",
                "WindDir (deg)",
                "WindVel (m/s)",
                "Longitude",
                "Latitude",
                "period_end",
                "prediction_date",
            ],
        ],
        road,
    )

    # drop prediction time col before joining
    solcast_prediction_time = solcast.prediction_date.max()
    solcast.drop(columns=["prediction_date"], inplace=True)
    tomorrow_prediction_time = tomorrow.prediction_date.max()
    tomorrow.drop(columns=["prediction_date"], inplace=True)

    forecast = solcast.loc[
        :,
        [
            "Distance (km)",
            "DirectSun (W/m2)",
            "DiffuseSun (W/m2)",
            "AirPress (Pa)",
            "AirTemp (degC)",
            "WindDir (deg)",
            "WindVel (m/s)",
            "Latitude",
            "Longitude",
        ],
    ].astype(np.float64)
    forecast = (
        forecast.reset_index()
        .set_index(["Distance (km)", "period_end"])
        .sort_values(by=["Distance (km)", "period_end"])
    )

    # TODO: insert the sunrise and sunset row before interpolate
    locations = get_locations(300)
    weather = tp.SSWeather()
    weather.data = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                locations["Distance (km)"],
                pd.date_range(race_start, race_end, freq="15min"),
            ],
            names=["Distance (km)", "period_end"],
        )
    )

    solcast["WindDirX"] = np.sin(solcast["WindDir (deg)"] / 180 * np.pi)
    solcast["WindDirY"] = np.cos(solcast["WindDir (deg)"] / 180 * np.pi)
    # interpolate to the desired resolution
    frame = pd.DataFrame(
        index=pd.date_range(race_start, race_end, freq="15min"),
        columns=locations["Distance (km)"],
    )

    def interp(param_name, df):
        parameter = df.pivot(columns="Distance (km)", values=param_name)

        frame.index.name = "period_end"
        parameter.dropna(inplace=True)
        parameter_interp = interpolate.interp2d(
            parameter.columns, parameter.index.astype(np.int64), parameter.to_numpy()
        )
        frame.loc[:, :] = parameter_interp(
            frame.columns.to_numpy(),
            frame.index.astype(np.int64).to_numpy(),
        )
        return (
            frame.melt(ignore_index=False, value_name=param_name)
            .reset_index()
            .set_index(["Distance (km)", "period_end"])
            .sort_values(by=["Distance (km)", "period_end"])
        )

    for param_name in [
        "DirectSun (W/m2)",
        "DiffuseSun (W/m2)",
        "AirTemp (degC)",
        "AirPress (Pa)",
        "WindVel (m/s)",
        "dni10",
        "dni90",
        "dhi10",
        "dhi90",
        "WindDirX",
        "WindDirY",
    ]:
        logging.debug("Interperting %s", param_name)
        out = interp(param_name, solcast)
        weather.data = weather.data.merge(out, on=["Distance (km)", "period_end"])

    # Pad to 0km and 3030km if not present
    if not (0.0 == weather.data.index.levels[0]).max():
        spot_forecast = weather.data.loc[forecast.index.levels[0].unique().min()]
        spot_forecast.loc[:, "Distance (km)"] = 0.0
        spot_forecast = spot_forecast.reset_index().set_index(
            ["Distance (km)", "period_end"]
        )
        weather.data = pd.concat([weather.data, spot_forecast])

    if not (3030.0 == weather.data.index.levels[0]).max():
        spot_forecast = weather.data.loc[forecast.index.levels[0].unique().max()]
        spot_forecast.loc[:, "Distance (km)"] = 3030.0
        spot_forecast = spot_forecast.reset_index().set_index(
            ["Distance (km)", "period_end"]
        )
        weather.data = pd.concat([weather.data, spot_forecast])

    weather.data = weather.data.reset_index()
    weather.data["WindDir (deg)"] = np.mod(
        np.arctan2(
            weather.data["WindDirX"].astype(np.float64),
            weather.data["WindDirY"].astype(np.float64),
        )
        / np.pi
        * 180,
        360,
    )
    weather.data = weather.data.merge(
        locations[["Distance (km)", "Latitude", "Longitude"]], on=["Distance (km)"]
    )
    weather.data = weather.data.set_index("period_end")
    missing = weather.data.loc[weather.data.isna().sum(axis=1) > 0]
    if missing.index.shape != (0,):
        # TODO: determin what to do when there is missing data.
        logging.warning(
            "%s missing values present in "
            "output weather file. Turn logging to debug for more details.",
            missing.isna().sum().sum(),
        )
        for i, row in missing.iterrows():
            logging.debug("Missing data at %s %s", i, row.index[row.isna()].to_list())

    weather.data = calc_sunpos(weather.data)

    weather.data.sort_values(by=["Distance (km)", "period_end"], inplace=True)
    if race_start.tzinfo != race_end.tzinfo:
        logging.warning(
            "Starting and ending time zone mismatch, "
            "using starting timezone as output timezone."
        )
    output_timezone = race_start.tzinfo
    weather.data = weather.data.tz_convert(output_timezone)
    weather.add_day_time_cols()

    weather.data.reset_index(inplace=True)
    weather.data.drop(columns="period_end", inplace=True)
    weather.zone.nj = weather.data.loc[:, "Distance (km)"].nunique()
    weather.zone.ni = weather.data.iloc[:, 0].count() / weather.zone.nj

    weather.data = weather.data[[
        "Day",
        "Time (HHMM)",
        "Distance (km)",
        "DirectSun (W/m2)",
        "DiffuseSun (W/m2)",
        "SunAzimuth (deg)",
        "SunElevation (deg)",
        "AirTemp (degC)",
        "AirPress (Pa)",
        "WindVel (m/s)",
        "WindDir (deg)",
        "dni10",
        "dni90",
        "dhi10",
        "dhi90",
        "Latitude",
        "Longitude",
    ]]

    weather.check_rectangular()

    # add tecplot title inc forecast dates
    weather.title = (
        f'WSC forecast starting at {race_start.strftime("%Y-%m-%d")} '
        f"localised to UTC{race_start.utcoffset().seconds / 3600:+}, "
        f'solcast SDK forecast from {solcast_prediction_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}, '  # pylint: disable=line-too-long
        f'tomorrow forecast from {tomorrow_prediction_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}, '  # pylint: disable=line-too-long
        f'generate at {datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}'
        # pylint: disable=line-too-long
    )
    return weather


def main(event, context):  # pylint: disable=unused-argument
    """Entry function from lambda functions on AWS.

    Args:
        event: event object from AWS (unused)
        context: context object from AWS (unused)

    Returns:
        None
    """
    logging.info("lambda function started")
    s3 = boto3.client("s3")
    # TODO: edit these before uploading to aws
    # Solcast only do prediction up to 7 day
    tz = pytz.timezone("Australia/Darwin")
    # race_start = tz.localize(datetime(2023, 9, 12))
    # race_end = tz.localize(datetime(2023, 9, 17))
    today = datetime.combine(datetime.today().date(), datetime.min.time())
    race_start = tz.localize(today + timedelta(1))
    race_end = tz.localize(today + timedelta(14))
    startime = race_start - timedelta(3)
    output_file = "/tmp/Weather-DEV2.dat"

    def partition_filter(x):
        return tz.localize(pd.Timestamp(x["prediction_date"])) > startime

    tomorrow = wr.s3.read_parquet(
        "s3://duscweather/tomorrow/",
        dataset=True,
        partition_filter=partition_filter,
    )

    # keep the useful columns only to conserve memory
    tomorrow = tomorrow.loc[
        :,
        [
            "temperature",
            "pressureSurfaceLevel",
            "windSpeed",
            "windDirection",
            "latitude",
            "longitude",
            "prediction_date",
            "period_end",
        ],
    ]

    solcast = wr.s3.read_parquet(
        "s3://duscweather/solcast_SDK/",
        dataset=True,
        partition_filter=partition_filter,
    )

    tomorrow.prediction_date = pd.to_datetime(tomorrow.prediction_date.astype(str))
    solcast.prediction_date = pd.to_datetime(solcast.prediction_date.astype(str))

    # tomorrow = pd.read_parquet("tomorrow.parquet")
    # solcast = pd.read_parquet("solcast.parquet")
    road = tp.TecplotData(r"RoadFile-LatLon-2021.dat")
    road_df = road.data

    weather = combine_forecast(solcast, tomorrow, road_df, race_start, race_end)
    logging.info("Writing tecplot...")
    weather.write_tecplot(output_file)
    logging.info("Uploading tecplot...")
    #  write to S3 and overwrite the latest
    wr.s3.upload(
        output_file,
        f"s3://duscweather/weather_files"
        f"{datetime.now(tz=timezone.utc).strftime('%Y-%m')}/Weather-"
        f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')}.dat",
    )
    wr.s3.upload(
        output_file,
        f"s3://duscweather/weather_files"
        f"{datetime.now(tz=timezone.utc).strftime('%Y-%m')}/Weather-latest.dat",
    )
    response = s3.put_object_acl(  # pylint: disable=unused-variable
        ACL="public-read",
        Bucket="duscweather",
        Key=f"weather_files{datetime.now(tz=timezone.utc).strftime('%Y-%m')}/"
        f"Weather-latest.dat",
    )
    logging.info("lambda function finished")


if __name__ == "__main__":
    main(0, 0)
