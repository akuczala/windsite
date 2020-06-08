import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry.linestring import LineString
from geopy.distance import distance
import transdist as this
#import requests
#from shapely.geometry import Point

MILE_FEET = 5280
MAX_LAT = 34.472499 #southern CA
#FAR = (-1000,-100)

def get_bounds(x):
    return x.bounds

#prepare / filter data

#shape file
trans_gdf = gpd.read_file('data/ca-transmission-lines/c1ba3265-9d1d-4aa2-88d2-d8e27e2d9e76202048-1-1beav9v.4src.shx')
trans_gdf['length'] = pd.to_numeric(trans_gdf['Length_Fee'])
trans_valid_gdf = trans_gdf[['kV_Sort','length','geometry']].dropna()
print(len(trans_valid_gdf))

trans_long_gdf = trans_valid_gdf[trans_valid_gdf['length'] > MILE_FEET*5]
print(len(trans_long_gdf))

special_gdf = trans_valid_gdf[(trans_valid_gdf['kV_Sort'] > 100) & (trans_valid_gdf['length'] > MILE_FEET*5)].reset_index()
print(len(special_gdf))
special_gdf['bounds'] = pd.DataFrame(special_gdf['geometry'].apply(get_bounds).to_numpy().copy(),columns=['bounds'])

special_gdf = special_gdf[special_gdf['bounds'].apply(lambda x: x[1] < MAX_LAT)]
print(len(special_gdf))


def sq_dist(lonlat1,lonlat2): #should use geopy dist here
    return (lonlat1[0]-lonlat2[0])**2 + \
    (lonlat1[1]-lonlat2[1])**2
def find_closest_in_geo(geo,lonlat):
    #closest = FAR
    if type(geo) == LineString:
        best_idx = np.argmin([sq_dist(lonlat,p) for p in geo.coords])
        return geo.coords[int(best_idx)]
    else:
        closest_coords = [
            find_closest_in_geo(line,lonlat) for line in geo
        ]
        best_idx = int(np.argmin([sq_dist(lonlat,p) for p in closest_coords]))
        return closest_coords[best_idx]
def find_closest_trans_point(latlon):
    lonlat = latlon[::-1]
    closest_coords = [find_closest_in_geo(this.special_gdf['geometry'].iloc[i],lonlat)
                     for i in range(len(this.special_gdf))]
    best_idx = int(np.argmin([sq_dist(lonlat,p) for p in closest_coords]))
    closest_point = closest_coords[best_idx][::-1]
    dist_miles = distance(latlon,closest_point).miles
    return (best_idx,closest_point,dist_miles) #convert back to latlon
latlon = (33.5,-115.5)