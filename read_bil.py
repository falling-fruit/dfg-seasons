import numpy as np

def read_prism_hdr(hdr_path):
    """Read an ESRI BIL HDR file"""
    with open(hdr_path, 'r') as input_f:
        header_list = input_f.readlines()
    return dict(item.strip().split() for item in header_list)
 
def read_prism_bil(bil_path):
    """Read an array from ESRI BIL raster file"""
    hdr_dict = read_prism_hdr(bil_path.replace('.bil', '.hdr'))
    # For now, only use NROWS, NCOLS, and NODATA
    # Eventually use NBANDS, BYTEORDER, LAYOUT, PIXELTYPE, NBITS
 
    prism_array = np.fromfile(bil_path, dtype=np.float32)
    prism_array = prism_array.reshape(
        int(hdr_dict['NROWS']), int(hdr_dict['NCOLS']))
    prism_array[prism_array == float(hdr_dict['NODATA'])] = np.nan
    return prism_array

bil_path = 'data_processing/PRISM_tmean_stable_4kmM3_199101_bil.bil'

prism_array = read_prism_bil(bil_path)
print(prism_array)

unique, counts = np.unique(prism_array, return_counts=True)
print(dict(zip(unique, counts)))


 


