import logging
import awswrangler as wr
import pandas as pd
from S5.Weather.solcast_forecast import send_request

logging.getLogger().setLevel(logging.INFO)
logging.info("lambda function started")
from config import solcast_api_key

API_KEY = solcast_api_key

locations = [
    [51.178882, -1.826215, "Stonehenge"],
    [41.89021, 12.492231, "The Colosseum"],
    [-12.4239, 130.8925, "Darwin_Airport"],
    [-19.64, 134.18, "Tennant_Creek_Airport"],
    [-23.7951, 133.8890, "Alice_Springs_Airport"],
    [-29.03, 134.72, "Coober_Pedy_Airport"],
    [-34.9524, 138.5196, "Adelaide_Airport"],
    [-31.1558, 136.8054, "Woomera"],
    [-16.262330910217, 133.37694753742824, "Daly_Waters"],
]

for location in locations:
    loc_df = send_request(location[0], location[1], API_KEY, location[2])
    try:
        df = pd.concat([df, loc_df], axis=0)
    except NameError:
        df = loc_df

wr.s3.to_parquet(
    df=df,
    path=f"s3://solcastresults/solcast/",
    dataset=True,
    mode="append",
    filename_prefix="solcast_",
)
