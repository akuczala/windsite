import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry.linestring import LineString
from geopy.distance import distance
import transdist as this
#import requests
#from shapely.geometry import Point

MILE_FEET = 5280
#MAX_LAT = 34.472499 #southern CA
MAX_LAT = 43 #all CA
#FAR = (-1000,-100)

def get_bounds(x):
    return x.bounds

def clean_ca(data_filename):
    #prepare / filter data

    #shape file
    trans_gdf = gpd.read_file(data_filename)
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
    this.special_gdf = special_gdf
    return this.special_gdf

def clean_tx(data_filename):
    #see power-lines-TX
    pass

def get_tx(data_filename):
    special_gdf = gpd.read_file(data_filename)
    special_gdf['radius'], special_gdf['center'] = zip(*special_gdf['geometry'].apply(calc_geo_circle))
    this.special_gdf = special_gdf

from shapely.geometry import LineString
from shapely.geometry.multilinestring import MultiLineString
def line_len(geo):
    if type(geo) == LineString:
        return sum(distance(lonlat1[::-1],lonlat2[::-1]).miles for lonlat1,lonlat2 in zip(test['geometry'].iloc[0].coords,geo.coords[1:]))
    if type(geo) == MultiLineString:
        return sum(line_len(gline) for gline in geo.geoms)
    print(type(geo))
    raise Exception('not a valid geometry')

def sq_dist(lonlat1,lonlat2): #should use geopy dist here
    return (lonlat1[0]-lonlat2[0])**2 + \
    (lonlat1[1]-lonlat2[1])**2
def distance_metric(lonlat1,lonlat2):
    return distance(lonlat1[::-1],lonlat2[::-1])
def distance_metric2(latlon1,latlon2):
    return distance(latlon1,latlon2).miles
def get_geo_bounds(geo):
    (min_lon,min_lat,max_lon,max_lat) = geo.bounds
    return ((min_lat,max_lat),(min_lon,max_lon))
def get_bounds_pts(bounds):
    return [
        (bounds[0][0],bounds[1][0]),
        (bounds[0][0],bounds[1][1]),
        (bounds[0][1],bounds[1][0]),
        (bounds[0][1],bounds[1][1])
        ]
def is_in_bbox(latlon,bounds):
    if bounds[0][0] < latlon[0] and latlon[0] < bounds[0][1]:
        if bounds[1][0] < latlon[1] and latlon[1] < bounds[1][1]:
            return True
        else:
            return False
    else:
        return False
def calc_geo_circle(geo):
    bpts = get_bounds_pts(get_geo_bounds(geo))
    center = (np.mean([bpt[0] for bpt in bpts]),np.mean([bpt[1] for bpt in bpts]))
    r = np.max([distance_metric2(bpt,center) for bpt in bpts])
    return r, center
def calc_circle_dist(latlon,r,center):
    return distance_metric2(latlon,center)-r
def bbox_is_in_radius(latlon,bounds,radius):
    pass
def find_closest_in_geo(geo,lonlat,max_dist = 50):
    #closest = FAR
    if type(geo) == LineString:
        best_idx = np.argmin([distance_metric2(lonlat,p) for p in geo.coords])
        return geo.coords[int(best_idx)]
    else:
        closest_coords = [
            find_closest_in_geo(line,lonlat) for line in geo
        ]
        best_idx = int(np.argmin([distance_metric2(lonlat,p) for p in closest_coords]))
        return closest_coords[best_idx]
def find_closest_in_linestring(latlon,geo):
    dists = [distance_metric2(latlon,glonlat[::-1]) for glonlat in geo.coords]
    best_idx = int(np.argmin(dists))
    min_dist = dists[best_idx]
    return geo.coords[best_idx][::-1], min_dist
def find_closest_in_geo_2(row,latlon,closest_dist,max_dist = 50):
    geo = row['geometry']
    r = row['radius']
    center = row['center']
    circ_dist = calc_circle_dist(latlon,r,center)
    if circ_dist < closest_dist:
        if type(geo) == LineString:
            return find_closest_in_linestring(latlon,geo)
        else: #MultiLineString
            closest_list = [
                find_closest_in_linestring(latlon,linestring) for linestring in geo
            ]
            best_idx = int(np.argmin([x[1] for x in closest_list]))
            return closest_list[best_idx]
    else:
            return (np.nan,np.nan), closest_dist
def find_closest_trans_point_2(latlon):
    closest_dist = 1e6
    closest_coord = (0,0)
    closest_i = -1
    #test = []
    #test2 = []
    for i in range(len(this.special_gdf)):
        coord, dist = find_closest_in_geo_2(this.special_gdf.iloc[i],latlon,closest_dist)
        if dist < closest_dist:
            closest_coord = coord
            closest_dist = dist
            closest_i = i
            #debug: return incremental closest points
            #test.append(closest_coord)
            #test2.append(closest_i)
    return (closest_i,closest_coord,closest_dist) #, test, test2
def find_closest_trans_point(latlon):
    lonlat = latlon[::-1]
    closest_coords = [find_closest_in_geo(this.special_gdf['geometry'].iloc[i],lonlat)
                     for i in range(len(this.special_gdf))]
    best_idx = int(np.argmin([distance_metric(lonlat,p) for p in closest_coords]))
    closest_point = closest_coords[best_idx][::-1]
    dist_miles = distance(latlon,closest_point).miles
    return (best_idx,closest_point,dist_miles) #convert back to latlon
#latlon = (33.5,-115.5)