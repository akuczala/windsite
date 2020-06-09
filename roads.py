import overpy
import geopy
import numpy as np

api = overpy.Overpass()

def make_query(lat,lon,radius=1000):
    return """<query type="way">
    <around lat=" """ + str(lat) + """ " lon=" """ + str(lon) + """ " radius=" """ + str(radius) + """ "/>
    </query>
    <union>
      <item/>
      <recurse type="down"/>
    </union>
    <print/>"""

from geopy.distance import distance
def sq_dist(latlon1,latlon2):
    return (latlon1[0]-latlon2[0])**2 + (latlon1[1]-latlon2[1])**2 
def nearest_node(latlon,way):
    dists = [distance(latlon,latlon_float((node.lat,node.lon))).miles for node in way.nodes]
    idx = np.argmin(dists)
    return (lambda n: latlon_float((n.lat,n.lon)))(way.nodes[idx]), np.min(dists)
def latlon_float(latlon):
    return (float(latlon[0]),float(latlon[1]))

#consider removing motorway, residential
def is_valid_highway(label):
    valid_highways = 'motorway,trunk,primary,secondary,tertiary,unclassified,residential'.split(',')
    return np.any([label == l for l in valid_highways]) 

def closest_valid_node(latlon,ways,debug=False):
		
    highway_labels = [way.tags.get("highway", "n/a") for way in ways]
    is_valid = list(map(is_valid_highway,highway_labels))
    valid_ways = [way for way,v in zip(ways,is_valid) if v]
    if debug:
    	print(len(valid_ways),'/',len(result.ways),'valid')

    if len(valid_ways) == 0:
        return (np.nan,np.nan), np.nan
    coords, dists = zip(*[nearest_node(latlon,way) for way in valid_ways])
    min_idx = np.argmin([dists])
    min_dist = dists[min_idx]
    min_coord = coords[min_idx]
    return min_coord, min_dist

def get_closest_road(latlon,debug=False):
	result = api.query(make_query(*latlon))
	return closest_valid_node(latlon,result.ways,debug=debug)