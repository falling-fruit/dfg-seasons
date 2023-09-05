# Develop for Good – Seasonality Prediction

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

# Data Pulling

## Overview
The following procedure describes the process for downloading phenology data from the following sources: Plant Phenology and PEP725. 

Plant Phenology (found [here](https://plantphenology.org/)) utilizes the USA National Phenology Network, which provides many phenology observations across the United States.
The Pan European Phenology Project PEP725 (found [here](http://www.zamg.ac.at/pep725/)) provides phenological data across Europe.

## Tools
[Plant Phenology](https://plantphenology.org/),  [PEP725](http://www.zamg.ac.at/pep725/), Microsoft Excel, Python

## Data Download - [Plant Phenology](https://pyphenology.readthedocs.io/en/master/data_structures.html)
First, search for the specific plant species you are interested in by their scientific name. Filter these results by _fruits present_, _ripening fruits present_, and _ripe fruits present_. 
At the top of the page, click the download button. The downloaded data does not include site IDs, so they must be generated.

Next, run `generate_unique_ids.py`. This will generate unique site IDs for each observation and output a new file containing 'site_id'.

## Data Download - [Pan European Phenology Project PEP725](http://www.zamg.ac.at/pep725/)
Users must have an account to download any data. On the homepage menu, navigate to the _data selection_ page. Using the dropdown menu, select the plant species of interest. 
The data is organized based on cultivar type and country. Click the download button for each country and cultivar type of interest. The downloaded data is not properly formatted for our needs, so it must be transformed.

In Microsoft Excel, first, navigate to 'Data' and click 'Get Data (Power Query)'. 
Next, click _Text/CSV_ to import data from the original downloaded csv file. Load the downloaded csv file titled “PEP725_XX_scientificName.csv”. Ensure the delimiter is _Semicolon_.
This should organize the data into proper columns.
Repeat this transformation process for the downloaded csv file titled “PEP725_XX_stations.csv”.

Open these new csv files, and using the BBCH codes provided in the original downloaded folder, determine which observations correlate with the presence of ripe fruit. Remove any rows that do not meet this criterion.
On the csv file, rename the BBCH code with the observation description (ex. BBCH code 86 -> "Ripe fruits")

Next, run `merge_pep.py`. This will merge the two datasets - “PEP725_XX_scientificName.csv” and “PEP725_XX_stations.csv” - and output a new file.
Open the new merged dataset and delete the following unnecessary columns - 'National _ID', 'ALT', and 'NAME'.

## Combining Datasets
Before merging datasets, ensure that all the datasets have identical column titles; otherwise, some data may be lost due to varying column titles.

DAY: the day of the year the observation was made
YEAR: the year the observation was made
genus: the taxonomic rank above species and below family; the first part of a scientific name for a species
specificEpithet: the second part of a scientific name, specifying the name of the species
eventRemarks: the observation description 
LAT: latitude; the distance north or south of the equator
LON: longitude; the distance east or west of the meridian
site_id: a unique identifier indicating the specific geographical location of an observation site 

Now, run `combine_datasets.py`. This will combine all the phenology datasets and export a new file.

Now this plant species phenology dataset is ready to be used for model training!

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

This file can also be downloaded from [here](https://drive.google.com/file/d/16OkaqtXTYisJDVMYl1pK_Mtz1NbvRiMG/view?usp=drive_link)

Next, download the monthly high-res weather data from this link: [monthly_weather_data.grib](https://drive.google.com/file/d/157JrxJsks9EmWLjyz1ERITu-Ds9mBo6M/view?usp=drive_link). 
Put the weather data in the following directory: `data/weather_data/monthly_weather_data.grib`. 

After this, you are all set to run the model training notebook!

For Monthly Data (the default, and best model), run all the cells in `Train from Monthly Data.ipynb`. 
This will train several models – one with all the plant observations, one that's aggressively filtered to only include observations in the "normal growing range", and one filtered to not include tropical observations. 

## Future Use

To train models using weather data from the future, you will first need a time machine. This can be found in `time_machine/blueprints_for_time_machine.py`. 
(Just kidding – there's no time machine).  

If you want to train/test these models again next year, you will need ERA-5 monthly data from the previous year. You will also need updated phenology observations. 
The documentation for these can be found at:



The model training notebook _should_ work on new data, as long as it is in the same format. You will need to update the high_cutoff_year global variable to 2023. If anything isn't working, feel free to email me at peterb2470@gmail.com or benson.p@northeastern.edu. 


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

