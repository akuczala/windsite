import streamlit as st
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt

st.title('WindSite')

DATA_DIR = 'test_opt_grid_vals_10x10.pkl'
FRAC_CAP_COL = 'frac_capacity'
CAPACITY_DEFAULT = 50
TRANS_COST_DEFAULT = 1000
TRANS_DIST_DEFAULT = 10
UTILITY_RATE_DEFAULT = 0.3
YEAR_KWH = 8.76e6
TRANS_DIST_COL = 'trans_dist'
N_SITES_DEFAULT = 3
@st.cache
def load_data():
    data = pd.read_pickle(DATA_DIR)
    wind_farm_data = pd.read_pickle('data/select-wind-power.pkl')
    wind_farm_data = wind_farm_data[[
    'elevation',
    'latitude',
    'longitude',
    'per_turb_cap_frac',
    'mean_wind_speed']]
    return data, wind_farm_data

#data_load_state = st.text('Loading data...')
data, wind_farm_data = load_data()
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
try:
	capacity = int(st.text_input('Capacity (MW)',CAPACITY_DEFAULT))
except:
	st.text('Invalid capacity')
	capacity = CAPACITY_DEFAULT

try:
	trans_cost= int(st.text_input('Transmission line $/mile',TRANS_COST_DEFAULT))
except:
	st.text('Invalid value')
	trans_cost = TRANS_COST_DEFAULT

utility_rate = text_box('Utility rate $/(kWh)',UTILITY_RATE_DEFAULT,dtype=float)

max_trans_dist = st.slider('Maximum distance to transmission line (Miles)', 0, 200, TRANS_DIST_DEFAULT)
max_n_sites = st.slider('Number of sites shown',1,10, N_SITES_DEFAULT, step=1)

filtered_data = data[data[TRANS_DIST_COL] <= max_trans_dist].sort_values(FRAC_CAP_COL,ascending=False).iloc[:max_n_sites]
st.subheader(f'Estimates for potential sites')

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
revenue = filtered_data[FRAC_CAP_COL]*YEAR_KWH*capacity*utility_rate

display_df = pd.DataFrame({
	'% Capacity' : filtered_data[FRAC_CAP_COL]*100,
 'Yearly revenue ($)' : revenue, 'Transmission Line Cost ($)' : trans_cost*filtered_data[TRANS_DIST_COL]}) 
display_df.index = np.arange(1,max_n_sites+1)
st.write(display_df)

st.map(filtered_data)