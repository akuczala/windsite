import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def clean_data(root_dir):
    data_arr = np.concatenate([
        np.load(root_dir + '/data/all-wind-1/features' + frange + '.npy')
	    for frange in '0:200,200:400,400:494,494:700,700:900,900:993'.split(',')
    ],axis=0)

    data_df = pd.read_pickle(root_dir + '/data/all-power-1-2.pkl').iloc[:len(data_arr)]
    data_df.columns = [' '.join(col).strip() for col in data_df.columns.values]
    data_df = data_df.reset_index().drop('index',axis=1)
    data_df = pd.concat([
        data_df,
        pd.DataFrame(data_arr,columns=['elevation','mean_wind_speed', 'pow_curve','temperature'])
    ],axis=1)

    data_df = data_df[(data_df['t_cap_factor'] < 1) & (data_df['p_cap_factor'] < 1)]
    #print(data_df.columns)

    usw_df = pd.read_csv(root_dir + '/data/uswtdbCSV/uswtdb_v3_0_1_20200514.csv')
    #data_df['t_capacity'] = ('t_cap<lambda>')
    #calc mean construction year
    data_df['p_year'] = [usw_df.set_index('eia_id').loc[ei,'p_year'].mean() for ei in data_df['eia_id']]
    #include state
    def extract_state(ei): #need this function because length-1 result has type str instead of series
        thing = usw_df.set_index('eia_id').loc[ei,'t_state']
        if type(thing) == str:
            return thing
        else:
            return thing.iloc[0]
    data_df['t_state'] = [extract_state(ei) for ei in data_df['eia_id']]

    clean_data_df = data_df[[
    'elevation','mean_wind_speed', 'pow_curve','temperature','p_year','t_state','eia_id',
    't_cap_factor','p_cap_factor',
    't_cap_factor_18','p_cap_factor_18','t_cap <lambda>','p_cap max','latitude','longitude']].dropna()

    return clean_data_df
    