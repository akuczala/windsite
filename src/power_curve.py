#defines interpolated power curve from NREL data,
#found here: https://www.nrel.gov/docs/fy16osti/66189.pdf (page 19)

from scipy.interpolate import interp1d
import pandas as pd

def get_power_curve():
	power_curve_df = pd.read_csv('../data/power-curves.csv')
	power_curve_fn = interp1d(power_curve_df['Speed'],power_curve_df['IEC - 3'],bounds_error=False,fill_value=0)
	return power_curve_fn