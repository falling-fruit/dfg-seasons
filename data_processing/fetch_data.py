#    #!/usr/bin/env python
# import cdsapi
# c = cdsapi.Client()
# c.retrieve('reanalysis-era5-land-monthly-means', { # Requests follow MARS syntax
#                                         # Keywords 'expver' and 'class' can be dropped. They are obsolete
#                                         # since their values are imposed by 'reanalysis-era5-complete'
#     'date'    : '2013-01-01',            # The hyphens can be omitted
#     'levelist': '1/10/100/137',          # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
#     'levtype' : 'ml',
#     'param'   : '130',                   # Full information at https://apps.ecmwf.int/codes/grib/param-db/
#                                         # The native representation for temperature is spherical harmonics
#     'stream'  : 'oper',                  # Denotes ERA5. Ensemble members are selected by 'enda'
#     'time'    : '00/to/23/by/6',         # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18'
#     'type'    : 'an',
#     'area'    : '80/-50/-25/0',          # North, West, South, East. Default: global
#     'grid'    : '1.0/1.0',               # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
#     'format'  : 'netcdf',                # Output needs to be regular lat-lon, so only works in combination with 'grid'!
# }, 'ERA5-ml-temperature-subarea.nc')     # Output file. Adapt as you wish.

import cdsapi

import pandas as pd

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'variable': [
            'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3',
            'soil_temperature_level_4',
        ],
        'year': '1950',
        'time': '00:00',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'format': 'grib',
        'product_type': 'monthly_averaged_reanalysis',
    },
    'download.grib')


# import cdsapi

# c = cdsapi.Client()

# c.retrieve(
#     'reanalysis-era5-land-monthly-means',
#     {
#         'product_type': 'monthly_averaged_reanalysis',
#         'variable': [
#             'lake_bottom_temperature', 'lake_ice_temperature', 'lake_mix_layer_temperature',
#             'lake_total_layer_temperature', 'soil_temperature_level_1', 'soil_temperature_level_2',
#             'soil_temperature_level_3', 'soil_temperature_level_4',
#         ],
#         'year': '1950',
#         'month': [
#             '01', '02', '03',
#             '04', '05', '06',
#             '07', '08', '09',
#             '10', '11', '12',
#         ],
#         'time': '00:00',
#         'format': 'grib',
#     },
#     'download.grib')

