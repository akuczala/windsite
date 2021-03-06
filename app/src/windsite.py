# The WindSite streamlit app, intended to be run with `streamlit run windsite.py`

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
#import st_state_patch #needed only when keeping track of state
import db
from db_names import *

#default values
TRANS_DIST_DEFAULT = 50
ROAD_DIST_DEFAULT = 6.0
RES_ROAD_DIST_DEFAULT = 1.5
N_SITES_DEFAULT = 3
#YEAR_KWH = 8.76e6

#settings
SQL_FILTER = True

st.title('WindSite')

#return the database connection
#cached so that we need only connect once
#must allow output mutation to use the connection
@st.cache(allow_output_mutation=True)
def get_database_connection():
    return db.connect()

#request data from the SQL connection
@st.cache(allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def load_data(max_trans_dist,max_road_dist,min_res_road_dist):

    conn = get_database_connection()

    if SQL_FILTER: #load only data that meet constraints
        data = db.load_with_constraints(conn,max_trans_dist,max_road_dist,min_res_road_dist)
    else: #load all data
        data = db.load_all_results(conn) 
    

    #filter out all sites estimated to have below average performance
    data = data[data[CAPACITY_CLASS_COL] > 0]

    return data

def text_box(label,default_val,error='Invalid value',dtype = int):
    try:
        val = st.text_input(label,default_val)
    except:
        st.text(error)
        val = default_val
    return dtype(val)


#--SIDEBAR, CONSTRAINTS--
st.sidebar.markdown('# Constraints')

show_sites_available = st.sidebar.empty()
max_trans_dist = st.sidebar.slider('Maximum distance to transmission line (miles)', 0, 50, TRANS_DIST_DEFAULT)
max_road_dist = st.sidebar.slider('Maximum distance to road (miles)', 0., 6.0, 6.0)
min_res_road_dist = st.sidebar.slider('Minimum distance to residential area (miles)', 0., 6., 1.5)

data = load_data(max_trans_dist,max_road_dist,min_res_road_dist)

if SQL_FILTER: #data already filtered
    filtered_data = data
else: #filter data with pandas
    filtered_data = data[
      (data[TRANS_DIST_COL] <= max_trans_dist) & \
      (data[ROAD_DIST_COL] <= max_road_dist) & \
      (data[RES_ROAD_DIST_COL] >= min_res_road_dist)
      ]

#display number of compatible sites
show_sites_available.markdown(str(len(filtered_data)) + ' compatible sites')

#adjust number of ranked sites
max_n_sites = st.sidebar.slider('Number of sites shown',1,10, N_SITES_DEFAULT, step=1)

#turn site heatmap on or off
show_heatmap = st.sidebar.checkbox('Show all compatible site locations', value=False)


#--SIDEBAR, WEIGHTS--
st.sidebar.markdown('# Weights')
st.sidebar.markdown('Least important --- Most important')

#weight sliders: format='' means that numeric values will not be shown
trans_weight = st.sidebar.slider('Transmission line distance', 0., 1., 0.5,key='a',format='')
road_weight = st.sidebar.slider('Road distance', 0., 1., 0.5,key='b',format='')
value_weight = st.sidebar.slider('Land price', 0., 1., 0.5,key='c',format='')

weights = dict(trans=trans_weight,road=road_weight,land_value = value_weight)

#calculate data ranges
@st.cache(show_spinner=False)
def get_ranges(data):
    return {
        'trans' : (np.min(data[TRANS_DIST_COL]),np.max(data[TRANS_DIST_COL])),
        'road' : (np.min(data[ROAD_DIST_COL]),np.max(data[ROAD_DIST_COL])),
        'land_value' : (np.min(data[LAND_VALUE_COL]),np.max(data[LAND_VALUE_COL])),
    }

#normalize data values to [0,1]
def range_scale(x,rng):
    return (x-rng[0])/(rng[1]-rng[0])

#normalize data values to [0,1] on a log scale
def log_range_scale(x,rng):
    log_x = np.log10(x)
    log_rng = tuple(map(np.log10,rng))
    return range_scale(log_x,log_rng)

#compute cost function for ranking: larger is worse
#the cost function is a convex sum of each factor
#since each factor is approximately log normal,
#the values are transformed to log space before taking the weighted sum
def cost_fn(values,weights,data_ranges):
    sum_weights = sum(weights[k] for k in ['trans','road','land_value'])
    return sum(weights[k]*log_range_scale(values[k],data_ranges[k]) for k in ['trans','road','land_value'])/sum_weights

#show number of sites limited by both the number of available sites, and user specification
n_shown_sites = min(len(filtered_data),max_n_sites)

#APP BODY
st.subheader(f'Estimates for potential sites')

#do not display dataframe if it is empty
if n_shown_sites == 0:
    st.write('No sites found')
    #dummy data to center map on texas
    map_df = pd.DataFrame({'lat' : [30], 'lon' : [-100], 'color' : [(0,0,0)]})
    #empty map layer
    map_layers = []
#display dataframe if it is nonempty
else: 
    #compute cost for each site
    data_ranges = get_ranges(filtered_data)
    filtered_data['costs'] = filtered_data.apply(
        lambda row: cost_fn(
            {
            'trans' : row[TRANS_DIST_COL], 'road' : row[ROAD_DIST_COL],'land_value' : row[LAND_VALUE_COL]
            },
            weights,
            data_ranges
        ), axis = 1
    )
    #extract top sites
    top_data = filtered_data.sort_values('costs',ascending=True).iloc[:n_shown_sites]

    n_stds = 2 #95% confidence interval
    price_lower = 10**(np.log10(top_data[LAND_VALUE_COL]) - n_stds*top_data[LAND_VALUE_LOGSTD_COL])
    price_upper = 10**(np.log10(top_data[LAND_VALUE_COL]) + n_stds*top_data[LAND_VALUE_LOGSTD_COL])
    display_df = pd.DataFrame({
        'Transmission line distance (miles)' : top_data[TRANS_DIST_COL],
        'Road distance (miles)' : top_data[ROAD_DIST_COL],
        #'Price per acre' : top_data[LAND_VALUE_COL],
        'Price per acre' : list(zip(price_lower,price_upper)),
        'NREL Capacity factor' : top_data[NREL_CF_COL],
        'County' : top_data[COUNTY_COL],
    })

    display_df.index = np.arange(1,n_shown_sites+1)
    #df.style.format({'B': "{:0<4.0f}", 'D': '{:+.2f}'})
    st.write(display_df.style.format({
        'Transmission line distance (miles)' : "{:0.2f}",
        'Road distance (miles)' :  "{:0.2f}",
        #'Price per acre' : "${:0.0f}",
        'Price per acre' : lambda lowhigh: "${0:0.0f}-{1:0.0f}".format(*lowhigh),
        #'NREL Capacity factor' : lambda x: "{0.2f}".format(x)
        #'+/-' : "${:0.2f}"
        #'County' : filtered_data[('nrel','County')],
    }))

    colors = [(255,0,0),(200,0,0),(150,0,0),(100,0,0)] + [(0,0,0) for _ in range(max(0,n_shown_sites-3))]

    map_df = pd.DataFrame({
        'lat' : top_data[NREL_LAT_COL].iloc[:n_shown_sites],
        'lon' : top_data[NREL_LON_COL].iloc[:n_shown_sites],
        'val' : top_data[LAND_VALUE_COL].iloc[:n_shown_sites],
        'site_number' : [str(x+1) for x in range(n_shown_sites)],
        'res_lat' : top_data[RES_ROAD_LAT_COL].iloc[:n_shown_sites],
        'res_lon' : top_data[RES_ROAD_LON_COL].iloc[:n_shown_sites],
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
        ),
        # pdk.Layer(
        #   'ScatterplotLayer',
        #   data=map_df,
        #   get_position='[res_lon, res_lat]',
        #   get_color = 'color',
        #   radiusScale = 50,
        # ),
    ]
    if show_heatmap:
        map_filter_df = pd.DataFrame({
            'lat' : filtered_data[NREL_LAT_COL],
            'lon' : filtered_data[NREL_LON_COL],
            })
        blue_map = [[241,238,246],[208,209,230],[166,189,219],[116,169,207],[43,140,190],[4,90,141]]
        transp_blue_map = [c + [200,] for c in blue_map]
        heatmap_layer = pdk.Layer(
            'HeatmapLayer',
            data=map_filter_df,
            get_position='[lon, lat]',
            color_range = transp_blue_map
            #get_color = '[0,255,0]',
            #radiusScale = 5000,
        )
        map_layers = [heatmap_layer,] + map_layers

st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/outdoors-v9',
     initial_view_state=pdk.ViewState(
         latitude= np.mean(map_df['lat']),
         longitude= np.mean(map_df['lon']),
         zoom=5,
         pitch=0,
     ),
     layers=map_layers
 ))