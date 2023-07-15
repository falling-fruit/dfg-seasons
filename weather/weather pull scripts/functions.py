import pandas as pd

def load_data(csv):
    df = pd.read_csv(csv)
    return df


def filter_by_long_lat(df, long, lat, delta_long, delta_lat):

    df2 = df.loc[df['Longtitude'] >= long - delta_long and df['Longtitude'] <= long + delta_long 
                 and df['Latitude'] >= lat - delta_lat and df['Latitude'] <= lat + delta_lat]
    
    ##return the desired rows(regions) in pandas dataframe 
    return df2

def find_unique_location_ids(df):

    # take three columns: location id, lat, long
    df = df[['location_id', 'lat', 'lng']]
    # filter by unique location id 
    df = df.drop_duplicates(subset=['location_id'])
    # use the first seen lat long for duplicate loc id
    return df




