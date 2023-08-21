# Develop for Good – Seasonality Prediction

### Model Training

This model requires historical weather data drawn from this google drive link:

https://drive.google.com/file/d/157JrxJsks9EmWLjyz1ERITu-Ds9mBo6M/view?usp=drive_link

Before training a model, make sure to download this into the data/ folder of this repo.

**Formatted Training Data:**
 
Weather data: https://drive.google.com/file/d/1nhahu2ei0QS9ITlGz2C_d4XvLfdXUCPH/view?usp=drive_link

Observation data: https://drive.google.com/file/d/16OkaqtXTYisJDVMYl1pK_Mtz1NbvRiMG/view?usp=drive_link

High-resolution weather data: https://drive.google.com/drive/folders/1nRuce11OdWAWnioUPt_C9Y9LBQp562AH?usp=drive_link

# Running Environment:
Package requirements:
```
- cfgrib
- xarray
- pandas
- numpy
- pyPhenology (may have to be installed with pip: https://pyphenology.readthedocs.io/en/master/install.html)
- matplotlib
- tqdm (helpful for model training progress)
```

Stock Python Packages:
```
- os
- warnings
- glob
```

The best approach to install and run these is to create a dedicated conda environment. Pandas has difficulty ensuring that all of the packages are compatible with each other. A tutorial on conda environments can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#).

# Claudia Data Pulling

# Data Formatting and Mode Training

## Overview & Special Considerations
pyPhenology requires a very strict data structure (found [here](https://pyphenology.readthedocs.io/en/master/data_structures.html)). It takes both plant data and weather data in the form of pandas dataframes. 
Crucially, these both need a "Site ID" column for this to function properly. 
Also, the Longitude coordinates in ERA-5 are **360-degree longitude**. I converted the longitude column from the plant observations, which is in 180 by default, to 360-degree. 

My approach here was to construct the site ID column based on the **Rounded Lat/Long Coordinates** of the sites. 
This means that the site IDs in the model training scripts **Do Not Match** with the Falling Fruit Site IDs. However, it would be fairly easy to match these based on coordinates. 
Here are some sample coordinates, and the pipeline to get to Site IDs:

(45.259, 222.885) -> (rounding step) -> (45.3, 222.9) -> (removing decimals) -> (453, 2229) -> (string concatenation) -> "4532229". 

These scripts assume you are running them from `modelling`. 

Many of the features used in the model training are stored in `modelling/model_training_helpers.py`. It is not necessary to have a complete understanding of everything going on in there, but I will give a brief summary at the end of this document. 

## Model Training Steps

First, run `construct_phenology_observations.py`. This will merge the data collected from NPN, PEP-5, and Falling Fruit into one dataframe. 
This will output a file to `data/model_training_data/all_plants_formatted.csv`. 

Next, download the monthly high-res weather data from this link: (monthly_weather_data.grib)[https://drive.google.com/file/d/157JrxJsks9EmWLjyz1ERITu-Ds9mBo6M/view?usp=drive_link]. 
Put the weather data in the following directory: `data/weather_data/monthly_weather_data.grib`. 

After this, you are all set to run the model training notebook!

For Monthly Data (the default, and best model), run all the cells in `Train from Monthly Data.ipynb`. 
This will train several models – one with all the plant observations, one that's aggressively filtered to only include observations in the "normal growing range", and one filtered to not include tropical observations. 

## Future Use

To train models using weather data from the future, you will first need a time machine. This can be found in `time_machine/blueprints_for_time_machine.py`. 
(Just kidding – there's no time machine).  

If you want to train/test these models again next year, you will need ERA-5 monthly data from the previous year. You will also need updated phenology observations. 
The documentation for these can be found at:



## Model Training Helpers

_In order of appearance in the script:_

The high cutoff year is set in this script as a global variable. This controls the slicing of test data. 

Next, there are error functions. MAE and RMSE are uesd to score the models. 

`make_test_df` creates a mock dataframe to be used for testing / model scoring. This is only used if there are no observations in the test year available. The mock dataframe consists of the mean observed ripeness day for the training data, but in 2022. This should give a good impression of the average ripeness day at that site. Howeve, there is some risk of overtraining with this approach. 

`train_ripeness_small` trains the ThermalTime model. Training is very easy, but testing is actually the difficult part. There are some considerations with missing data during testing, so the actual input to the predict function is filtered by site ID. Also, sometimes, the model returns a prediction of 999, so I just filter these out. Then, I print the model and score it. 

`claudia_observations_to_pyphenology` performs some basic formatting steps, but is used only in `construct_phenology_observations.py`. 

`correct_leap_years` maps all months onto the same 12 values. This effectively removes all leap years from the data. Only really works for monthly data. 

`format_weather_data` formats a dataframe converted straight from a GRIB file into a usable dataframe. Mostly this is basic date formatting stuff, column renaming, etc. 

`train_species_models`. This is the biggest function in the whole thing, combining all the other ones. This trains a unique model for each species. The biggest challenge is missing data. There are some types of missing data that are simply impossible to fix, like no training data or no weather data. 

One that is possible to fix is a lack of test data. To correct for this, I use the `make_test_df` function, which interpolates mock test sites. 

This function iterates through all species in the given plant dataframe, and returns two things: a list of models (list of dicts, with "species" and "model" fields. Useful for saving model parameters if desired.); and a dataframe containing predictions. This is processed during model scoring, but 

`score_model` prints various model error metrics. Number of species is useful for comparing how many species had enough data to train. For example, with European models, the number of species was often quite low (<10). 

MAE stands for Mean Absolute Error. The functions for this and RMSE can be found at the top. 

Median Absolute Error is a useful metric in this case. The absolute error is heavily right-skewed, meaning that there are many cases of small errors, and a few cases of large errors. The median error gives the threshold where 50% of predictions have an error less than the median. For example, a median of 10 would mean that 50% of the predictions are within 10 days of the observed value. 

SD portion represents the portion of plants that lie under 1 standard deviation of the data. 
This calculation is repeated for a 1-month threshold and a 2*SD threshold. 

For example, if the month portion was 0.9, it would mean that 90% of the predictions fell within 1 month of the observed date. 

