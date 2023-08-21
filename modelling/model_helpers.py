#import cfgrib
#import xarray as xr

import pandas as pd
import numpy as np

#from pyPhenology import models, utils

#from tqdm import trange, tqdm

#import matplotlib.pyplot as plt

#from warnings import warn
from pyPhenology import models, utils

import os, glob

from tqdm import trange, tqdm


### This file contains all functions necessary for formatting our data and processing it into a format for training. 

### CUTOFF YEAR = 2022
high_cutoff_year = 2022

def aic(obs, pred, n_param):
        return len(obs) * np.log(np.mean((obs - pred)**2)) + 2*(n_param + 1)

def rmse(y1, y2):
        return np.sqrt(np.mean((y1 - y2) ** 2))

def mae(y1, y2):
        return np.mean(np.abs(y1 - y2))

# Interpolate testing data from training data (mean)
def make_test_df(train_df):
    #print(train_df)
    species_sites = train_df['site_id'].unique()
        
    #print(species_sites)
    
    site_ripenesses = []

    for site in species_sites:
        site_df = train_df[train_df['site_id'] == site]

        site_ripenesses.append({
            'site_id': site,
            'doy': np.mean(site_df['doy'])
        })

    species_test_df = pd.DataFrame(site_ripenesses)
    species_test_df['year'] = high_cutoff_year
    
    species_test_df['formatted_sci_name'] = train_df['formatted_sci_name'].iloc[0]
    
    return species_test_df

# Training function specific to our uses.
def train_ripeness_small(observations, predictors, test_observations, test_predictors, model_name = 'ThermalTime'):

    print("running model {m}".format(m=model_name))
    Model = utils.load_model(model_name)
    model = Model()
    model.fit(observations, predictors, optimizer_params='practical')
    
    print(model)
    
    print("making predictions for model {m}".format(m=model_name))
    
    pred_list = []
    
    # assuming year is the same for both test obs and test preds
    for s in test_observations['site_id'].unique():
        site_obs = test_observations[test_observations['site_id'] == s]
        site_prediction = model.predict(site_obs, test_predictors)
        
        if len(site_prediction) > 0 and site_prediction[0] < 999:
            pred_list.append({
                'site_id': s,
                'formatted_sci_name': test_observations['formatted_sci_name'].iloc[0],
                'prediction': site_prediction[0],
                'doy': site_obs['doy'].iloc[0]
            })
        
    #print(pred_list)
    
    pred_df = pd.DataFrame.from_records(pred_list)
    
    #print(pred_df)
    
    preds = pred_df['prediction']
    test_days = pred_df['doy']
    sites = pred_df['site_id']

    if len(preds) == 0:
        print(test_observations)
        print(test_predictors)
    
    #print(pred_list)
   
    # Various error types
    model_mae = mae(test_days, preds)
    model_rmse = rmse(test_days, preds)
    median_error = np.median(np.abs(test_days - preds))

    print('model {m} got a MAE of {a}'.format(m=model_name,a=model_mae))
    print('model {m} got an RMSE of {a}'.format(m=model_name,a=model_rmse))
    print('model {m}\'s median error is: {a}'.format(m=model_name,a=median_error))

    print("Ripeness Day: {}".format(np.mean(preds)))
    
    #filtered_test_observations = test_observations[test_observations['site_id'].isin(sites)]
    
    ripeness_data = pred_df
    ripeness_data['ripeness_day'] = ripeness_data['prediction']
    
    ripeness_dict = {
        'model_object': model,
        'MAE': model_mae,
        'RMSE': model_rmse,
        'Median Error': median_error,
        'prediction_df': ripeness_data,
    }
    
    return model, ripeness_dict

# Format Claudia's Data
def claudia_observations_to_pyphenology(claudia_obs):
    new_observations = claudia_obs.copy(deep=True)
    
    new_observations['species_actual'] = new_observations['specificEpithet']
    
    new_observations.rename(columns={'YEAR': 'year',
                            'DAY': 'doy',
                            'genus': 'species',
                            'LAT': 'latitude'}, inplace=True)
    
    new_observations.drop(['specificEpithet', 'eventRemarks', 'LON', 'lon_360'], axis=1, inplace=True)
    
    new_observations['phenophase'] = 516
    
    return new_observations

def correct_leap_years(weather_df):
    leap_year_key = {60: 61, 
                 91: 92, 
                 121: 122, 
                 152: 153, 
                 182: 183, 
                 213: 214, 
                 244: 245, 
                 274: 275, 
                 305: 306, 
                 335: 336}
    
    return weather_df.replace({'doy': leap_year_key})

def format_weather_data(raw_grib_df, correct_leap_year=True):
    print("Converting GRIB to dataframe")
    new_df = raw_grib_df.drop(["number", "step", "surface"], axis=1).reset_index().rename(columns={"skt":"temperature"})
    
    print('formatting date columns')
    new_df['year'] = new_df.time.dt.to_period('Y').astype(str).astype(int)
    new_df['doy'] = new_df.time.dt.strftime('%j').astype(int)
    
    new_df = new_df[['temperature', 'year', 'doy', 'latitude', 'longitude']]
    
    if correct_leap_year:
        print("correcting leap years")
        new_df = correct_leap_years(new_df)

    print("rounding columns and constructing site ID")
    new_df['latitude'] = np.round(new_df['latitude'], 1)
    new_df['longitude'] = np.round(new_df['longitude'], 1)

    new_df['coordstring'] = new_df['latitude'].astype(str) + new_df['longitude'].astype(str)

    new_df['site_id'] = new_df['coordstring'].str.replace(".", "")
    
    return new_df


## THE TRAINING FUNCTION. 

# This function takes in your plant data and weather data and:
# Separates into training/testing
# Trains a unique model for each species. 
# If there isn't enough test data, test data is created by interpolating the mean ripeness of training data for each site. 
# Concatenates the predictions into one dataframe. 

# Train threshold is the minimum observations required to train a model of a given species.
# Test threshold is the same, but for test data. If this falls short, test data will be interpolated. 
def train_species_models(full_plant_data, full_weather_data, train_threshold=10, test_threshold=1, save_dir='trained_models'):
    trained_models = []
    
    # Separate weather data into train and test
    weather_training = full_weather_data[full_weather_data['year'] < high_cutoff_year]
    weather_test = full_weather_data[full_weather_data['year'] >= high_cutoff_year]
    
    species_prediction_dict = {}
    species_list = full_plant_data['formatted_sci_name'].unique()
    print(species_list)

    for s in tqdm(species_list):
        print("\n\n", s)
        species_train_df = full_plant_data.query('formatted_sci_name == "{}" and year < {}'.format(s, high_cutoff_year))
        
        #print(species_train_df)
        if len(species_train_df) < train_threshold:
            print("not enough training data")
            continue        
        
        species_test_df = full_plant_data.query('formatted_sci_name == "{}" and year >= {}'.format(s, high_cutoff_year))

        if len(species_test_df) < test_threshold:
            print("Not enough test data for {}, interpolating".format(s))
            # make predictions and compare to the mean ripeness day at each site
            species_test_df = make_test_df(full_plant_data)

        if len(species_test_df) == 0:
            print("No test data for {}, after attempt at rectification".format(s))
            #print(species_test_df)
        
        if len(weather_test) == 0:
            print("No weather data")
            continue
            
        filtered_weather_test = weather_test[weather_test['site_id'].isin(species_test_df['site_id'])]
        filtered_species_test = species_test_df[species_test_df['site_id'].isin(filtered_weather_test['site_id'])]
        
        #print(species_train_df, weather_training)
        #print(species_test_df)
        #print(filtered_weather_test)
        #print(np.sort(filtered_species_test['site_id'].unique()))
        #print(np.sort(filtered_weather_test['site_id'].unique()))
        
        print("Training Observations: ", len(species_train_df))
        print("Testing Observations: ", len(filtered_species_test))
        
        model, predictions = train_ripeness_small(species_train_df, weather_training,
                            filtered_species_test, filtered_weather_test)
        
        trained_models.append(model)
        
        # Save the model for later
        # model.save_params(os.path.join(save_dir, s))
        #save_model(model, s, save_dir) 

        #break
        
        species_prediction_dict[s] = predictions
    
    #print(species_prediction_dict)
        
    # gets a list of all the prediction dataframes from the species model
    df_list = [i['prediction_df'] for i in species_prediction_dict.values()]

    full_prediction_df = pd.concat(df_list)

    full_prediction_df['abs_error'] = np.abs(full_prediction_df['doy'] - full_prediction_df['ripeness_day'])
    
    #print(score_model(full_prediction_df))
    
    return trained_models, full_prediction_df

  
## STATS / MODEL SCORING

# Prints various error metrics. Takes in a prediction df (from train_species_models).
def score_model(prediction_df):
    observed = prediction_df['doy']
    predicted = prediction_df['ripeness_day']
    
    n_species = prediction_df['formatted_sci_name'].nunique()
    
    print("Number of Species:", n_species)
    
    median_err = np.round(np.median(prediction_df['abs_error']), 2)
    std = np.round(np.std(prediction_df['abs_error']), 2)
    
    # MAE, RMSE, median
    
    print("Error metrics:")
    print("MAE:", np.round(mae(observed, predicted), 2))
    print("RMSE:", np.round(rmse(observed, predicted), 2))
    print("Median Absolute Error:", np.round(median_err, 2))
    
    # portion of errors under the SD
    print("SD portion (SD = {})".format(std))
    print(np.round(len(prediction_df.query('abs_error < {}'.format(std))) / len(prediction_df), 2))
    
    print("Month threshold:")
    print(np.round(len(prediction_df.query('abs_error < 30')) / len(prediction_df), 2))

    # 
    print("2 * SD portion (2SD = {})".format(2 * std))
    print(np.round(len(prediction_df.query('abs_error < {}'.format(2 * std))) / len(prediction_df), 2))
    

# get how "good" one sample is compared to the whole sample
def calc_error_percentile(sample, full_sample):
    sample_median = np.median(sample['abs_error'])
    
    print(sample_median)
    print(1 - len(full_sample.query('abs_error < {}'.format(sample_median))) / len(full_sample))