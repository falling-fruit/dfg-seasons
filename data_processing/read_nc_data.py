from scipy.io import netcdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

file2read = netcdf.NetCDFFile('ERA5-ml-temperature-subarea.nc','r')
temp = file2read.variables[var] # var can be 'Theta', 'S', 'V', 'U' etc..
data = temp[:]*1
file2read.close()

plt.contourf(data[t,z,:,:])