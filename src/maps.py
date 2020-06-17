import geopandas as gpd

def get_usa_gdf():
	return gpd.read_file('../maps/states_21basic/states.shp')
def get_conusa_gdf():
	usa = get_usa_gdf()
	return usa[(usa.STATE_ABBR != 'HI') & (usa.STATE_ABBR != 'AK')]
def get_state_gdf(state_abbr):
	usa_gdf = get_usa_gdf()
	return usa_gdf[usa_gdf.STATE_ABBR == state_abbr]