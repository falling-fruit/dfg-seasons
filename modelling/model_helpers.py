import cfgrib
import xarray as xr

import pandas as pd
import numpy as np

from pyPhenology import models, utils

from tqdm import trange, tqdm

import matplotlib.pyplot as plt

from warnings import warn


default_models = [models.ThermalTime(), models.FallCooling(), models.M1(), models.MSB()]
default_model_names = ['ThermalTime', "FallCooling", "M1", "MSB"]

# Turns a dataframe containing predictions of flowering day to a dict? Not sure what this does. 
def ripeness_data_to_dict(ripeness_data):    
    
    mean_maturation = np.mean(ripeness_data['flowering_day'])
    
    prediction_dict = {
        "full_flowering_data": ripeness_data,
        #"species_site_flowering days": list(ripeness_data['flowering_day']),
        "mean_flowering_day": np.mean(ripeness_data['flowering_day'])
    }
    
    return prediction_dict


def aic(obs, pred, n_param):
        return len(obs) * np.log(np.mean((obs - pred)**2)) + 2*(n_param + 1)


# Trains a model with a given set of test observations and test predictors. 
def train_ripeness(observations, predictors, test_observations, test_predictors, models=['ThermalTime']):
    # set up model comparisons
    best_aic=np.inf
    best_model = None
    best_model_name = None

    # iterate through all models
    for model_name in models:
        print("running model {m}".format(m=model_name))
        
        Model = utils.load_model(model_name)
        model = Model()
        model.fit(observations, predictors, optimizer_params='practical')
        
        # predict from test observations
        print("making predictions for model {m}".format(m=model_name))        
        preds = model.predict(test_observations, test_predictors)
        
        #print(preds)
        test_days = test_observations.doy.values
        #print(test_days)
        # this isn't valid - need to filter by site IDs
        
        # THIS IS REALLY BAD:
        test_days = test_days[0:len(preds)]
        #print(test_days)
        
        # score model
        model_aic = aic(obs = test_days,
                        pred=preds,
                        n_param = len(model.get_params()))

        if model_aic < best_aic:
            best_model = model
            best_model_name = model_name
            best_aic = model_aic

        print('model {m} got an aic of {a}'.format(m=model_name,a=model_aic))

    print('Best model: {m}'.format(m=best_model_name))
    print('Best model paramters:')
    print(best_model.get_params())
    print("Ripeness Day: {}".format(np.mean(preds)))
    
    ripeness_data = test_observations
    ripeness_data['flowering_day'] = preds
    
    return ripeness_data

# Trains a model and uses a portion of the training data for testing. 
def train_ripeness_percent(observations, predictors, test_percent, models=['ThermalTime']):
    test_observations = observations.sample(frac=test_percent)
    observations_train = observations.drop(test_observations.index)
    
    # set up model comparisons
    best_aic=np.inf
    best_model = None
    best_model_name = None

    # iterate through all models
    for model_name in models:
        print("running model {m}".format(m=model_name))
        
        Model = utils.load_model(model_name)
        model = Model()
        model.fit(observations_train, predictors, optimizer_params='practical')
        
        # predict from test observations
        print("making predictions for model {m}".format(m=model_name))        
        preds = model.predict(test_observations, predictors)
    
        #print(preds)
        test_days = test_observations.doy.values
        #print(test_days)
        
        # THIS IS REALLY BAD:
        test_days = test_days[0:len(preds)]
        #print(test_days)
        
        # score model
        model_aic = aic(obs = test_days,
                        pred=preds,
                        n_param = len(model.get_params()))
        print(model_aic)

        if model_aic < best_aic:
            best_model = model
            best_model_name = model_name
            best_aic = model_aic

        print('model {m} got an aic of {a}'.format(m=model_name,a=model_aic))

    print('Best model: {m}'.format(m=best_model_name))
    print('Best model paramters:')
    print(best_model.get_params())
    print("Ripeness Day: {}".format(np.mean(preds)))
    
    ripeness_data = test_observations
    ripeness_data['flowering_day'] = preds
    
    return ripeness_data


# Gets the weather history for a specific site. 
def get_site_history(weather_array, site_id, site_lat, site_lon):
    filtered = weather_array.where((abs(weather_array.latitude - site_lat) <= 0.05) & (abs(weather_array.longitude - site_lon) <= 0.05), drop=True)
    
    #print("Converting GRIB to dataframe")
    site_df = filtered.to_dataframe().drop(["number", "step", "surface"], axis=1).reset_index().rename(columns={"skt":"temperature"})
    
    site_df['site_id'] = site_id
    
    site_df['year'] = site_df.time.dt.to_period('Y')
    site_df['doy'] = site_df.time.dt.strftime('%j').astype(int)
    
    site_df = site_df[['site_id', 'temperature', 'year', 'doy', 'latitude', 'longitude']]
    
    return(site_df)

def get_site_history_coarse(weather_array, site_id, site_lat, site_lon):
    filtered = weather_array.where((abs(weather_array.latitude - site_lat) <= 0.5) & (abs(weather_array.longitude - site_lon) <= 0.5), drop=True)
    
    #print("Converting GRIB to dataframe")
    site_df = filtered.to_dataframe().drop(["number", "step", "surface"], axis=1).reset_index().rename(columns={"skt":"temperature"})
    
    site_df['site_id'] = site_id
    
    site_df['year'] = site_df.time.dt.to_period('Y')
    site_df['doy'] = site_df.time.dt.strftime('%j').astype(int)
    
    site_df = site_df[['site_id', 'temperature', 'year', 'doy', 'latitude', 'longitude']]
    
    return(site_df)

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