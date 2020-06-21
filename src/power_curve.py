#defines interpolated power curve from NREL data,
#found here: https://www.nrel.gov/docs/fy16osti/66189.pdf (page 19)

from scipy.interpolate import interp1d
import pandas as pd

def get_power_curve(col):
	power_curve_df = pd.read_csv('../data/power-curves.csv')
	power_curve_fn = interp1d(power_curve_df['Speed'],power_curve_df[col],bounds_error=False,fill_value=0)
	return power_curve_fn

def get_power_curve_num(n):
	if n == 0: #code offshore as 0
		return get_power_curve('Offshore')
	col = 'IEC - ' + str(n)
	return get_power_curve(col)