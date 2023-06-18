import pandas as pd

def load_data(csv):
    df = pd.read_csv('temp.csv')
    return df


def filter_by_long_lat(df, long, lat, delta_long, delta_lat):

    df2 = df.loc[df['Longtitude'] >= long - delta_long and df['Longtitude'] <= long + delta_long 
                 and df['Latitude'] >= lat - delta_lat and df['Latitude'] <= lat + delta_lat]
    
    ##return the desired rows(regions) in pandas dataframe 
    return df2

