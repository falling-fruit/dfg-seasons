import cfgrib
import xarray as xr

import pandas as pd
import numpy as np

from pyPhenology import models, utils

from tqdm import trange, tqdm

import matplotlib.pyplot as plt

from warnings import warn

from daylength import daylength

import os, glob

cutoff_year = 2010

# Format Claudia's Data
def claudia_observations_to_pyphenology(claudia_obs):
    new_observations = claudia_obs.copy(deep=True)
    
    new_observations['species'] = new_observations['specificEpithet']
    
    new_observations.rename(columns={'YEAR': 'year',
                            'DAY': 'doy',
                            'LAT': 'latitude'}, inplace=True)
    
    new_observations.drop(['specificEpithet', 'eventRemarks', 'LON'], axis=1, inplace=True)
    
    new_observations['phenophase'] = 516
    
    return new_observations

### Load all plant csvs
# assumes running from the modelling directory.
print("Finding Plant Data")
path = os.getcwd()
parent_dir = os.path.dirname(path)
#print(parent_dir)

final_path = os.path.join(parent_dir, "data/plant phenology/final fruit datasets/*.csv")
#print(final_path)

csv_files = glob.glob(final_path)
#print(csv_files)

# Merge plant data
print("Merging Claudia Plant Data") 

plant_data_list = []

for f in csv_files:
    df = pd.read_csv(f)
    
    plant_data_list.append(df)
    
final_plant_data = pd.concat(plant_data_list)

# format plant data
print('Formatting Longitude, Filtering Years')
#print(final_plant_data.columns)

final_plant_data["lon_360"] = final_plant_data["LON"] % 360
formatted_plants = claudia_observations_to_pyphenology(final_plant_data)
#print(formatted_plants['genus'][0:10])
formatted_plants = formatted_plants[formatted_plants['year'] >= cutoff_year].drop_duplicates()


# Construct species column
formatted_plants['species'] = formatted_plants.species.fillna('')

formatted_plants['sci_name'] = formatted_plants['genus'] + " " + formatted_plants['species']

### Format Species (drop cultivar)

# leaving this here cause i'm proud of it
#regex_pattern = r"('.*'|\([\w\s]+\)| x |sub[g|sp]\. [a-z]+|-[a-z]+| s[\w\s]+)" 

print('Filtering out Subspecies')
# filter out special characters(like that one species)
formatted_plants['formatted_sci_name'] = formatted_plants['sci_name']

single_quote_pattern = r"'.*'"
formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.replace(single_quote_pattern, "", regex=True)

paren_pattern = r"\([\w\s]+\)"
formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.replace(paren_pattern, "", regex=True)

subspecies_pattern = r"s[a-z]+\.( [\w]+|)"
formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.replace(subspecies_pattern, "", regex=True)

x_pattern = r" x( [a-z]+|)"
formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.replace(x_pattern, "", regex=True)

dash_pattern = r"-[\w]+"
formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.replace(dash_pattern, "", regex=True)

cultivar_pattern = r" (early cultivar|late cultivar)"
formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.replace(cultivar_pattern, "", regex=True)

special_cases = r"(Adirondack|Novosadski)"
formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.replace(special_cases, "", regex=True)

formatted_plants['formatted_sci_name'] = formatted_plants['formatted_sci_name'].str.strip()

## Done formatting subspecies

formatted_plants.drop('sci_name', axis=1, inplace=True)

formatted_plants.to_csv("../data/model_training_data/all_plants_formatted.csv")
