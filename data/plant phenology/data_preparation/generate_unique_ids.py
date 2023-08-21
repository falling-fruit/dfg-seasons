#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd


# # Generate unique site ids

# In[17]:


def generate_unique_ids(input_file, output_file, unique_id_column_name, column1_index, column2_index):
    unique_ids = {}
    rows = []  
    
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        headers.append(unique_id_column_name)
        
        for row in reader:
            column1 = row[column1_index]
            column2 = row[column2_index]
            
            # Combine the values from column1 and column2
            combined_value = str(column1) + str(column2)
            
            # Check if the combined value already has an ID assigned
            if combined_value in unique_ids:
                # Use the existing ID
                unique_id = unique_ids[combined_value]
            else:
                # Generate a new ID
                unique_id = str(len(unique_ids) + 1)
                unique_ids[combined_value] = unique_id
            
            row.append(unique_id)
            rows.append(row)  
    
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)  
    
# Read the generated CSV file as a DataFrame
df = pd.read_csv(output_file)

# Print the new CSV column
print(df[unique_id_column_name].values)
print(df.head())

# Usage example
generate_unique_ids('original.csv', 'new.csv', 'site_id', 5, 6)


# In[ ]:




