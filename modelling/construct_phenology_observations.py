import cfgrib
import xarray as xr

import pandas as pd
import numpy as np

from pyPhenology import models, utils

from tqdm import trange, tqdm

import matplotlib.pyplot as plt

from warnings import warn

from daylength import daylength

# Format Claudia's Data
def claudia_observations_to_pyphenology(claudia_obs):
    new_observations = claudia_obs.copy(deep=True)
    
    new_observations['species_actual'] = new_observations['specificEpithet']
    
    new_observations.rename(columns={'YEAR': 'year',
                            'DAY': 'doy',
                            'genus': 'species',
                            'LAT': 'latitude'}, inplace=True)
    
    new_observations.drop(['specificEpithet', 'eventRemarks', 'LON'], axis=1, inplace=True)
    
    new_observations['phenophase'] = 516
    
    return new_observations

### Load all plant csvs
# assumes running from the modelling directory.
path = os.getcwd()
parent_dir = os.path.dirname(path)
#print(parent_dir)

final_path = os.path.join(parent_dir, "data/plant phenology/final fruit datasets/*.csv")
#print(final_path)

csv_files = glob.glob(final_path)
#print(csv_files)

# Merge plant data
plant_data_list = []

for f in csv_files:
    df = pd.read_csv(f)
    
    plant_data_list.append(df)
    
final_plant_data = pd.concat(plant_data_list)

# format plant data
final_plant_data["lon_360"] = final_plant_data["LON"] % 360
formatted_plants = claudia_observations_to_pyphenology(final_plant_data)
formatted_plants = formatted_plants[formatted_plants['year'] >= cutoff_year].drop_duplicates()

filtered_observations.to_csv("../data/model_training_data/all_plants_formatted.csv")
