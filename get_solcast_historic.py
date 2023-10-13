"""Get forecast from solcast and stores it on S3"""
import logging
import time

import awswrangler as wr
import dask
import pandas as pd
from config import solcast_api_keys
from utils import get_locations
import os
from solcast import live
import numpy as np
import datetime

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

s5_logger = logging.getLogger("S5")
s5_logger.setLevel(logging.DEBUG)
s5_logger.addHandler(handler)

logging.info("lambda function started")


def main(event, context):  # pylint: disable=unused-argument
    os.environ["SOLCAST_API_KEY"] = solcast_api_keys[0]
    # Read road file and get the locations
    locations = get_locations(300)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    period = "PT5M"
    df = []
    for _, location in locations.iterrows():
        loc_df = dask.delayed(get_spot_live)(location, period, timestamp)
        df.append(loc_df)

    df = dask.compute(*df)
    df2 = pd.concat(df)
    print(df2["latitude"].nunique())
    wr.s3.to_parquet(
        df=df2,
        path="s3://duscweather/solcast_SDK/",
        dataset=True,
        mode="append",
        filename_prefix="solcast_",
        partition_cols=["prediction_date"],
    )

def get_spot_live(location, period, timestamp):
    retries = 3
    for i in range(retries):
        res = live.radiation_and_weather(
            latitude=location["Latitude"],
            longitude=location["Longitude"],
            output_parameters=[
                "dni",
                "dhi",
                "air_temp",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "azimuth",
                "zenith",
            ],
            period=period,
            hours=168,
        )
        if res.code == 200:
            logging.debug("successful request for %s", ["Name"])
            loc_df = res.to_pandas()
            loc_df = loc_df.rename(
                columns={
                    "surface_pressure": "pressureSurfaceLevel",
                    "wind_speed_10m": "windSpeed",
                    "wind_direction_10m": "windDirection",
                }
            )
            loc_df.reset_index(inplace=True)
            loc_df.loc[:, "period"] = period
            loc_df.loc[:, "period"] = loc_df.loc[:, "period"].astype(
                pd.CategoricalDtype()
            )
            loc_df["latitude"] = location["Latitude"]
            loc_df["longitude"] = location["Longitude"]
            loc_df["location_name"] = location["Name"]
            loc_df["prediction_date"] = np.datetime64(
                pd.Timestamp(timestamp))
            return loc_df
        elif res.code == 429:
            logging.info(
                "rate limited for request %s, %d retry remaining, "
                "retrying in %d seconds",
                location["Name"],
                retries - i,
                int(res.exception[-10:-8]) + 1,
            )
            time.sleep(int(res.exception[-10:-8]) + 1)
    logging.error("Unable to get forecast for %s, error %s",
                  location["Name"],
                  res.exception)


if __name__ == "__main__":
    main(0, 0)
