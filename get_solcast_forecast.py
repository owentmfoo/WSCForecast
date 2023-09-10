"""Get forecast from solcast and stores it on S3"""
import logging
import awswrangler as wr
import pandas as pd
from S5.Weather.solcast_forecast import send_request
from config import solcast_api_keys
from utils import get_locations, test_locations

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

logging.info("lambda function started")
test_locations = test_locations[:0]


def main(event, context):  # pylint: disable=unused-argument
    # Read road file and get the locations
    locations = get_locations()

    df = None
    for i, key in enumerate(solcast_api_keys):
        for location in test_locations:
            loc_df = send_request(location[0], location[1], key, location[2])
            if loc_df.shape == (0, 0):
                logging.error("Bad Solacast API Key %s", key)
            df = pd.concat([df, loc_df], axis=0)
        for _, location in locations.iloc[10 * i: 10 * (i + 1), :].iterrows():
            loc_df = send_request(
                location["Latitude"], location["Longitude"], key, location["Name"]
            )
            df = pd.concat([df, loc_df], axis=0)

    wr.s3.to_parquet(
        df=df,
        path="s3://duscweather/solcast/",
        dataset=True,
        mode="append",
        filename_prefix="solcast_",
        partition_cols=["prediction_date"],
    )
