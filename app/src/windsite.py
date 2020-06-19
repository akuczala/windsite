import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
#from matplotlib import pyplot as plt

st.title('WindSite')

DATA_DIR = 'data/results.pkl'
FRAC_CAP_COL = 'frac_capacity'
CAPACITY_DEFAULT = 50
TRANS_COST_DEFAULT = 1000
TRANS_DIST_DEFAULT = 10
UTILITY_RATE_DEFAULT = 0.3
YEAR_KWH = 8.76e6
TRANS_DIST_COL = ('trans','distance')
ROAD_DIST_COL = ('road','distance')
LAND_VALUE_COL = ('ML','land_value')
N_SITES_DEFAULT = 3
@st.cache
def load_data():
    data = pd.read_pickle(DATA_DIR)
    #filter out everything with the bad capacity class
    data = data[data[('ML','capacity_class')] > 0]
    # wind_farm_data = pd.read_pickle('data/select-wind-power.pkl')
    # wind_farm_data = wind_farm_data[[
    # 'elevation',
    # 'latitude',
    # 'longitude',
    # 'per_turb_cap_frac',
    # 'mean_wind_speed']]
    return data #, wind_farm_data

#data_load_state = st.text('Loading data...')
data = load_data()
#data_load_state.text("Done")

def text_box(label,default_val,error='Invalid value',dtype = int):
	try:
		val = st.text_input(label,default_val)
	except:
		st.text(error)
		val = default_val
	return dtype(val)
# st.subheader('percent power vs wind')

# if st.checkbox('Show plot'):
# 	plt.scatter(data['mean_wind_speed'],data['per_turb_cap_frac'])
# 	st.pyplot()

# st.button('Update')

#capacity_filter = st.slider('Capacity (MW)', 0, 200, 50)
# try:
# 	capacity = int(st.text_input('Capacity (MW)',CAPACITY_DEFAULT))
# except:
# 	st.text('Invalid capacity')
# 	capacity = CAPACITY_DEFAULT

# try:
# 	trans_cost= int(st.text_input('Transmission line $/mile',TRANS_COST_DEFAULT))
# except:
# 	st.text('Invalid value')
# 	trans_cost = TRANS_COST_DEFAULT

#utility_rate = text_box('LMP or PPA $/(kWh)',UTILITY_RATE_DEFAULT,dtype=float)

max_trans_dist = st.sidebar.slider('Maximum distance to transmission line (Miles)', 0, 200, TRANS_DIST_DEFAULT)
max_road_dist = st.sidebar.slider('Maximum distance to road (Miles)', 0., 5., 0.2)
max_n_sites = st.sidebar.slider('Number of sites shown',1,10, N_SITES_DEFAULT, step=1)

def weight_label(x):
	if x == 1:
		return 'Important'
	if x == 0:
		return 'Unimportant'
	return x
trans_weight = st.sidebar.slider('Transmission weight', 0., 1., 0.5, format = weight_label(x))
road_weight = st.sidebar.slider('Road weight', 0., 1., 0.5)
value_weight = st.sidebar.slider('Land price weight', 0., 1., 0.5)
weights = dict(trans=trans_weight,road=road_weight,land_value = value_weight)

def get_ranges(data):
	return {
    	'trans' : (np.min(data[TRANS_DIST_COL]),np.max(data[TRANS_DIST_COL])),
    	'road' : (np.min(data[ROAD_DIST_COL]),np.max(data[ROAD_DIST_COL])),
    	'land_value' : (np.min(data[LAND_VALUE_COL]),np.max(data[LAND_VALUE_COL])),
    }
def range_scale(x,rng):
	return (x-rng[0])/(rng[1]-rng[0])
def cost_fn(values,weights,data_ranges):
	sum_weights = sum(weights[k] for k in ['trans','road','land_value'])
	return sum(weights[k]*range_scale(values[k],data_ranges[k]) for k in ['trans','road','land_value'])/sum_weights
 
filtered_data = data[(data[TRANS_DIST_COL] <= max_trans_dist) & (data[ROAD_DIST_COL] <= max_road_dist)]
#st.write(filtered_data)
data_ranges = get_ranges(filtered_data)
costs = filtered_data.apply(
	lambda row: cost_fn(
		{
		'trans' : row[TRANS_DIST_COL], 'road' : row[ROAD_DIST_COL],'land_value' : row[LAND_VALUE_COL]
		},
		weights,
		data_ranges
	), axis = 1
)

n_shown_sites = min(len(filtered_data),max_n_sites)


st.subheader(f'Estimates for potential sites')
if n_shown_sites == 0:
	st.write('No sites found')
	map_df = pd.DataFrame({'lat' : [30], 'lon' : [-100], 'color' : [(0,0,0)]})
	map_layers = []
else:
	filtered_data['costs'] = costs

	filtered_data = filtered_data.sort_values('costs',ascending=True).iloc[:n_shown_sites]
	display_df = pd.DataFrame({
		'Transmission line distance (miles)' : filtered_data[TRANS_DIST_COL],
		'Road distance (miles)' : filtered_data[ROAD_DIST_COL],
		'Price per acre' : filtered_data[LAND_VALUE_COL],
		#'Price per acre' : 10**(np.log10(filtered_data[LAND_VALUE_COL]) - filtered_data[('ML','land_value_logstd')]),
		#'+/-' : 10**(np.log10(filtered_data[LAND_VALUE_COL]) + filtered_data[('ML','land_value_logstd')]),
		'County' : filtered_data[('nrel','County')],
	})

	display_df.index = np.arange(1,n_shown_sites+1)
	#df.style.format({'B': "{:0<4.0f}", 'D': '{:+.2f}'})
	st.write(display_df.style.format({
		'Transmission line distance (miles)' : "{:0.2f}",
		'Road distance (miles)' :  "{:0.2f}",
		'Price per acre' : "${:0.0f}",
		#'+/-' : "${:0.2f}"
		#'County' : filtered_data[('nrel','County')],
	}))

	colors = [(255,0,0),(200,0,0),(150,0,0),(100,0,0)] + [(0,0,0) for _ in range(max(0,n_shown_sites-3))]

	map_df = pd.DataFrame({
		'lat' : filtered_data[('nrel','latitude')].iloc[:n_shown_sites],
		'lon' : filtered_data[('nrel','longitude')].iloc[:n_shown_sites],
		'val' : filtered_data[('ML','land_value')].iloc[:n_shown_sites],
		'site_number' : [str(x+1) for x in range(n_shown_sites)],
		'color' : colors[:n_shown_sites]
	})
	map_layers=[
		pdk.Layer(
			'TextLayer',
			data=map_df,
			get_position='[lon, lat]',
			get_color = 'color',
			get_text='site_number',
			get_size=30,
			get_angle=0,
			get_text_anchor='middle',
		)]
# import gmaps
# import os
# import geopandas as gpd
# gmaps.configure(api_key=os.environ["GOOGLE_API_KEY"])
# point_list = list(zip(filtered_data['latitude'],filtered_data['longitude']))
# point_layer = gmaps.symbol_layer(
#     point_list, fill_color=(0,255,0), stroke_color=(0,255,0), scale=2
# )
# fig = gmaps.figure(
#     layout={
#         'width': '800px',
#         'height': '600px',
#     },
#     map_type='SATELLITE')
# fig.add_layer(point_layer)
# fig
#st.subheader('Estimated revenue / year')
# revenue = filtered_data[FRAC_CAP_COL]*YEAR_KWH*capacity*utility_rate

# display_df = pd.DataFrame({
# 	'% Capacity' : filtered_data[FRAC_CAP_COL]*100,
#  'Yearly revenue ($)' : revenue, 'Transmission Line Cost ($)' : trans_cost*filtered_data[TRANS_DIST_COL]}) 

st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/outdoors-v9',
     initial_view_state=pdk.ViewState(
         latitude= np.mean(map_df['lat']),
         longitude= np.mean(map_df['lon']),
         zoom=5,
         pitch=0,
     ),
     layers=map_layers
		# pdk.Layer(
		# 	'ScatterplotLayer',
		# 	data=map_df,
		# 	get_position='[lon, lat]',
		# 	get_color = 'color',
		# 	radiusScale = 500,
		# ),
         # pdk.Layer(
         #    'ColumnLayer',
         #    data=map_df,
         #    get_position='[lon, lat]',
         #    radius=2000,
         #    get_fill_color = '[48, 255, 40]',
         #    get_elevation='val',
         #    elevation_scale=10,
         #    elevation_range=[0, 1000],
         #    pickable=True,
         #    extruded=True,
         # ),
 ))