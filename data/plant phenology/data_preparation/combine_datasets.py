#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd 
import numpy as np


# # Combine Datasets from Various Sources

# In[ ]:


# Read the first dataset
data1 = pd.read_csv("/falling fruit/data1.csv") # Replace with appropriate filepath 

# Read the second dataset
data2 = pd.read_csv("/falling fruit/data2.csv") # Replace with appropriate filepath 

# Concat the datasets
fruit = pd.concat([data1, data2], axis = 0)

# Export to a new CSV file
fruit.to_csv('/falling fruit/fruit.csv', index = False)

# Display the new dataset 
print(fruit)

