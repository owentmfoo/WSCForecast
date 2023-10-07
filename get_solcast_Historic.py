"""Get forecast from solcast and stores it on S3"""
import logging
import awswrangler as wr
import pandas as pd
from S5.Weather.solcast_forecast import send_request
from config import solcast_api_keys
from utils import get_locations, test_locations
import os
from solcast import live

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
    locations = get_locations(1200)
    # locations = test_locations
    df = None
    for location in locations:
        res = live.radiation_and_weather(
            latitude=location[0],
            longitude=location[1],
            output_parameters=["dni", "dni10", "dni90",
                               "dhi", "dhi10", "dhi90",
                               "temperature",
                               "surface_pressure",
                               "wind_speed_10m",
                               "windDirection",
                               "azimuth","zenith"]
        )
        try:
            loc_df = res.to_pandas()
            loc_df.rename({"surface_pressure": "pressureSurfaceLevel",
                           "wind_speed_10m": "windSpeed",
                           "wind_direction_10m": "windDirection"})
            df = pd.concat([df, loc_df], axis=0)
        except Exception as e:
            logging.exception(e)

    wr.s3.to_parquet(
        df=df,
        path="s3://duscweather/solcast/",
        dataset=True,
        mode="append",
        filename_prefix="solcast_",
        partition_cols=["prediction_date"],
    )


if __name__ == '__main__':
    main(0, 0)
