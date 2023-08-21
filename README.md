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

pyPhenology requires a very strict data structure (found [here](https://pyphenology.readthedocs.io/en/master/data_structures.html)). It takes both plant data and weather data in the form of pandas dataframes. 
Crucially, these both need a "Site ID" column for this to function properly. 
Also, the Longitude coordinates in ERA-5 are **360-degree longitude**. I converted the longitude column from the plant observations, which is in 180 by default, to 360-degree. 
My approach here was to construct the site ID column based on the **Rounded Lat/Long Coordinates** of the sites. 
This means that the site IDs in the model training scripts **Do Not Match** with the Falling Fruit Site IDs. However, it would be fairly easy to match these based on coordinates. 
Here are some sample coordinates, and the pipeline to get to Site IDs:

- (45.259, 222.885) -> (rounding step) -> (45.3, 222.9) -> (removing decimals) -> (453, 2229) -> (string concatenation) -> "4532229". 

