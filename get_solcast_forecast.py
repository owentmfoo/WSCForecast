"""Get forecast from solcast and stores it on S3"""
import logging
import warnings

import awswrangler as wr
import pandas as pd
from S5.Weather.solcast_forecast import send_request
from config import solcast_api_keys
from utils import get_locations, test_locations
import os
from solcast import forecast
import numpy as np
import datetime
import tqdm

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
warnings.filterwarnings("ignore",category=DeprecationWarning)

def main(event, context):  # pylint: disable=unused-argument
    os.environ["SOLCAST_API_KEY"] = solcast_api_keys[0]
    # Read road file and get the locations
    locations = get_locations(300)
    # locations = test_locations
    df = None
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    period = 'PT5M'
    for _,location in tqdm.tqdm(locations.iterrows(),total=locations.shape[0]):
        res = forecast.radiation_and_weather(
            latitude=location["Latitude"],
            longitude=location["Longitude"],
            output_parameters=["dni", "dni10", "dni90",
                               "dhi", "dhi10", "dhi90",
                               "air_temp",
                               "surface_pressure",
                               "wind_speed_10m",
                               "wind_direction_10m",
                               "azimuth","zenith"],
            period=period,
            hours=336
        )
        try:
            loc_df = res.to_pandas()
            loc_df = loc_df.rename(columns={"surface_pressure": "pressureSurfaceLevel",
                           "wind_speed_10m": "windSpeed",
                           "wind_direction_10m": "windDirection",
                           })
            loc_df.reset_index(inplace=True)
            loc_df.loc[:, "period"] = period
            loc_df.loc[:, "period"] = loc_df.loc[:, "period"].astype(
                pd.CategoricalDtype())
            loc_df["latitude"] = location["Latitude"]
            loc_df["longitude"] = location["Longitude"]
            loc_df["location_name"] = location["Name"]
            loc_df["prediction_date"] = np.datetime64(pd.Timestamp(timestamp))
            df = pd.concat([df, loc_df], axis=0)
        except Exception as e:
            logging.exception(e)

    wr.s3.to_parquet(
        df=df,
        path="s3://duscweather/solcast_SDK/",
        dataset=True,
        mode="append",
        filename_prefix="solcast_",
        partition_cols=["prediction_date"],
    )


if __name__ == '__main__':
    main(0, 0)
