#utilities for reading country and state geodataframes

import geopandas as gpd

#load all states
def get_usa_gdf():
	return gpd.read_file('../maps/states_21basic/states.shp')
#load continental USA
def get_conusa_gdf():
	usa = get_usa_gdf()
	return usa[(usa.STATE_ABBR != 'HI') & (usa.STATE_ABBR != 'AK')]
#load a specific state by its state code
def get_state_gdf(state_abbr):
	usa_gdf = get_usa_gdf()
	return usa_gdf[usa_gdf.STATE_ABBR == state_abbr]
#returns gdf with a state's bounding box - the state's shape
#useful for plotting heatmaps on state
def get_state_mask_gdf(state_abbr):
	state = get_state_gdf(state_abbr)
	state_box = gpd.GeoDataFrame({'geometry':state.envelope})
	anti_state = gpd.overlay(state_box,state,how='difference')
	return anti_state