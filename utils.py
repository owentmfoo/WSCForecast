"""utility code from getting forecast data"""
from config import solcast_api_keys
from S5 import Tecplot as tp
import pandas as pd
import numpy as np

test_locations = [
    [51.178882, -1.826215, "Stonehenge"],
    [41.89021, 12.492231, "The Colosseum"],
]


def get_locations(n_points=10 * len(solcast_api_keys)):
    road = tp.TecplotData(r"RoadFile-LatLon-2021.dat")
    locations = pd.DataFrame(
        np.linspace(0, 3030, n_points), columns=["Target Distance"]
    )
    locations.loc[:, ["Distance (km)", "Latitude", "Longitude"]] = np.nan

    for i, spot in locations[["Target Distance"]].iterrows():
        data = road.data.loc[
            abs(road.data["Distance (km)"] - spot["Target Distance"]).argmin(),
            ["Distance (km)", "Latitude", "Longitude"],
        ]
        locations.loc[i, ["Distance (km)", "Latitude", "Longitude"]] = data

    locations["Name"] = "WSC" + locations["Distance (km)"].astype(str) + "km"
    return locations
