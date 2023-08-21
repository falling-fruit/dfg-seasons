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


## Model Training Helpers

If you want to train/test these models again next year, you will need ERA-5 monthly data from the previous year. You will also need updated phenology observations. 
The documentation for these can be found at:


