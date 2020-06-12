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

def clean_data():
    data_arr = np.concatenate([
        np.load('../data/all-wind-1/features' + frange + '.npy')
	    for frange in '0:200,200:400,400:494,494:700,700:900,900:993'.split(',')
    ],axis=0)

    data_df = pd.read_pickle('data/all-power-1.pkl').iloc[:len(data_arr)]
    data_df.columns = [' '.join(col).strip() for col in data_df.columns.values]
    data_df = data_df.reset_index().drop('index',axis=1)
    data_df = pd.concat([
        data_df,
        pd.DataFrame(data_arr,columns=['elevation','mean_wind_speed', 'pow_curve','temperature'])
    ],axis=1)

    data_df = data_df[(data_df['t_cap_factor'] < 1) & (data_df['p_cap_factor'] < 1)]
    data_df = data_df[data_df['p_cap max'] > 10]

    usw_df = pd.read_csv('../data/uswtdbCSV/uswtdb_v3_0_1_20200514.csv')
    #calc mean construction year
    data_df['p_year'] = [usw_df.set_index('eia_id').loc[ei,'p_year'].mean() for ei in data_df['eia_id']]

    clean_data_df = data_df[['elevation','mean_wind_speed', 'pow_curve','temperature','t_cap_factor','p_cap_factor','p_year']].dropna()

    return clean_data_df
    