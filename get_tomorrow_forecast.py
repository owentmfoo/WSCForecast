"""Get forecast from tommorrow.io and stores it on S3"""
import logging
import awswrangler as wr
import pandas as pd
from S5.Weather.tomorrow_forecast import send_request
from config import tomorrow_api_keys
from utils import get_locations, test_locations

logging.getLogger().setLevel(logging.INFO)
logging.info("lambda function started")
test_locations = test_locations[:0]


def main(event, context):  # pylint: disable=unused-argument
    # Read road file and get the locations
    locations = get_locations()

    df = None
    for _, key in enumerate(tomorrow_api_keys):
        for location in test_locations:
            loc_df = send_request(location[0], location[1], key, location[2])
            if loc_df.shape == (0, 0):
                logging.error("Bad tomorrow API Key %s", key)
            df = pd.concat([df, loc_df], axis=0)
        for _, location in locations.iterrows():
            loc_df = send_request(
                location["Latitude"], location["Longitude"], key, location["Name"]
            )
            df = pd.concat([df, loc_df], axis=0)

    wr.s3.to_parquet(
        df=df,
        path="s3://duscweather/tomorrow/",
        dataset=True,
        mode="append",
        filename_prefix="tomorrow_",
        partition_cols=["prediction_date"],
    )
