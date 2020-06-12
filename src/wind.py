import h5pyd
import os
import requests
import numpy as np
import pandas as pd
import dateutil
from pyproj import Proj #for elevation, index lookup
from IPython.display import display
import wind as this#self import

##taken from NREL github
# This function finds the nearest x/y indices for a given lat/lon.
# Rather than fetching the entire coordinates database, which is 500+ MB, this
# uses the Proj4 library to find a nearby point and then converts to x/y indices
def indicesForCoord(f, lat_index, lon_index):
    dset_coords = f['coordinates']
    projstring = """+proj=lcc +lat_1=30 +lat_2=60 
                    +lat_0=38.47240422490422 +lon_0=-96.0 
                    +x_0=0 +y_0=0 +ellps=sphere 
                    +units=m +no_defs """
    projectLcc = Proj(projstring)
    origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
    origin = projectLcc(*origin_ll)
    
    coords = (lon_index,lat_index)
    coords = projectLcc(*coords)
    delta = np.subtract(coords, origin)
    ij = [int(round(x/2000)) for x in delta]
    return tuple(reversed(ij))

#set up NREL 
def setup_nrel():
	filename = "/nrel/wtk-us.h5"
	f = h5pyd.File(filename, 'r')
	dset = f['windspeed_100m']
	dt = get_datetimes(f)
	return dict(f =f, dset = dset, dt= dt)

#get datetimes
def get_datetimes(file):
	dt = file["datetime"]
	dt = pd.DataFrame({"datetime": dt[:]},index=range(0,dt.shape[0]))
	dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)
	return dt

#indices for each year
def get_year_idx(dt,yr):
    return dt.loc[(dt.datetime >= str(yr) + '-01-01') & (dt.datetime < str(yr + 1) + '-01-01')].index

def get_speeds(dset,loc_idx,time_idxs):
    #print(type(time_idxs))
    if type(time_idxs) == str:
        if time_idxs == 'all':
            return dset[:, loc_idx[0], loc_idx[1]]
        if time_idxs == 'test':
            return dset[::100, loc_idx[0], loc_idx[1]]
    else:
        return dset[min(time_idxs):max(time_idxs)+1, loc_idx[0], loc_idx[1]]

def get_vals(dset,loc_idx,time_idxs):
    #print(type(time_idxs))
    if type(time_idxs) == str:
        if time_idxs == 'all':
            return dset[:, loc_idx[0], loc_idx[1]]
        if time_idxs == 'test':
            return dset[::100, loc_idx[0], loc_idx[1]]
    else:
        return dset[min(time_idxs):max(time_idxs)+1, loc_idx[0], loc_idx[1]]
def get_vals_at(dset,loc_idx_list,time_idxs):
    return [get_vals(dset,loc_idx,time_idxs) for loc_idx in loc_idx_list]


#from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import ElasticNetCV
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import cross_val_score

#df = pd.read_pickle('data/select-wind-power.pkl')
#wind_speeds = np.load('data/farm_wind_speeds_060320.npy')

this.features = ['elevation','latitude','longitude','mean_wind_speed','mean_cubed_wind_speed']

def fit_model(wind_farm_data,features,score = 'r2', random_state = 0):
	X = wind_farm_data[features]
	y = wind_farm_data['per_turb_cap_frac']

	model = LinearRegression()

	score_funs = {
	'explained_variance' : explained_variance_score,
	'r2' : r2_score
	}
	score_fun = score_funs[score]

	#split the data into training / testing set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= random_state)

	#cross validate with f1 scores
	scores = cross_val_score(model,X_train,y_train,scoring = score)

	print('cross validation scores:', scores)

	#fit data and compare scores for training and testing data
	model = model.fit(X_train, y_train)

	print("training data")
	y_pred = model.predict(X_train)
	print(score_fun(y_train, y_pred))

	print("testing data")
	y_pred = model.predict(X_test)
	print(score_fun(y_test, y_pred))

	return model

def calc_elevation(lat, lng):
	apikey = os.environ["GOOGLE_API_KEY"]
	url = "https://maps.googleapis.com/maps/api/elevation/json"
	request = requests.get(url+"?locations="+str(lat)+","+str(lng)+"&key="+apikey)
	try:
		results = request.json().get('results')
		if 0 < len(results):
			elevation = results[0].get('elevation')
			# ELEVATION
			return elevation
		else:
			print('HTTP GET Request failed.')
	except ValueError:
		print('JSON decode failed: '+str(request))

from scipy.interpolate import interp1d
power_curve_df = pd.read_csv('data/power-curves.csv')
power_curve_fn = interp1d(power_curve_df['Speed'],power_curve_df['IEC - 3'],bounds_error=False,fill_value=0)

def latlon_to_features(nrel,latlon,features,year = 2013):
	lat, lon = latlon

	elevation = calc_elevation(lat,lon)

	year_idx = get_year_idx(nrel['dt'],year)
	loc_idx = indicesForCoord(nrel['f'],lat,lon)

	wind_speeds =  get_speeds(nrel['dset'],loc_idx,year_idx)
	mean_wind_speed = np.mean(wind_speeds)
	#mean_cubed_wind_speed = np.mean(wind_speeds**3)
	pow_curve = np.mean(power_curve_fn(wind_speeds))
	mean_temp, mean_pressure, mean_precip = [
	    np.mean(get_vals(nrel['f'][dset_name],loc_idx,year_idx))
	    for dset_name in ['temperature_80m','pressure_100m','precipitationrate_0m']
	]

	featmap = {'latitude' :lat, 'longitude' : lon, 'elevation' : elevation,
	'mean_wind_speed' : mean_wind_speed, 'pow_curve' : pow_curve,
	'temperature': mean_temp, 'pressure' : mean_pressure, 'precipitation' : mean_precip}

	x = np.array([featmap[feat] for feat in features])

	return x

def idx_to_features(nrel,coords,loc_idx,features,year = 2013):
	lat, lon = coords[loc_idx]

	elevation = calc_elevation(lat,lon)

	year_idx = get_year_idx(nrel['dt'],year)

	wind_speeds =  get_speeds(nrel['dset'],loc_idx,year_idx)
	mean_wind_speed = np.mean(wind_speeds)
	#mean_cubed_wind_speed = np.mean(wind_speeds**3)
	pow_curve = np.mean(power_curve_fn(wind_speeds))
	mean_temp = np.mean(get_vals(nrel['f']['temperature_80m'],loc_idx,year_idx))

	featmap = {'latitude' :lat, 'longitude' : lon, 'elevation' : elevation,
	'mean_wind_speed' : mean_wind_speed, 'pow_curve' : pow_curve,
	'temperature': mean_temp}

	x = np.array([featmap[feat] for feat in features])

	return x

def idx_to_feature_history(nrel,coords,loc_idx,features,year = 2013):
	lat, lon = coords[loc_idx]

	elevation = calc_elevation(lat,lon)

	year_idx = get_year_idx(nrel['dt'],year)

	wind_speeds =  get_speeds(nrel['dset'],loc_idx,year_idx)
	mean_wind_speed = np.mean(wind_speeds)
	#mean_cubed_wind_speed = np.mean(wind_speeds**3)
	pow_curve = np.mean(power_curve_fn(wind_speeds))
	mean_temp = np.mean(get_vals(nrel['f']['temperature_80m'],loc_idx,year_idx))

	featmap = {'latitude' :lat, 'longitude' : lon, 'elevation' : elevation,
	'mean_wind_speed' : mean_wind_speed, 'pow_curve' : pow_curve,
	'temperature': mean_temp}

	x = np.array([featmap[feat] for feat in features])

	return x

