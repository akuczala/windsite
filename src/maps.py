import geopandas as gpd

def get_usa_gpd():
	return gpd.read_file('maps/states_21basic/states.shp')
def get_state_gpd(state_abbr):
	usa_gpd = get_usa_gpd()
	return usa_gpd[usa_gpd.STATE_ABBR == state_abbr]