#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd 


# # PEP725 - Merging Phenology and Location Data 

# In[ ]:


# Read the CSV file
phenology = pd.read_csv("/PEP725/PEP725_AT_Prunus_avium(early_cultivar).csv") # Replace with appropriate filepath 
stations = pd.read_csv("/PEP725/PEP725_AT_stations.csv") # Replace with appropriate filepath 

# Merge the two datasets on column 'PEP_ID'
merged_data = pd.merge(phenology, stations, on = 'PEP_ID')

# Export to a new CSV file
merged_data.to_csv('/falling fruit/merged_data.csv', index = False) # Replace with appropriate filepath 

# Display the new merged dataset
print(merged_data)

