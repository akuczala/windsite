import pandas as pd
import re
import geopandas as gpd
import numpy as np
import maps
from matplotlib import pyplot as plt

import imp
imp.reload(maps)
import maps

def scrape_landwatch():
	pass

def parse_text(text):
    try:
        split = re.split('Acres|,|\n|\$',text)
        return (float(split[0]),split[1],split[2],int(''.join(split[4:])))
    except:
        #raise
        print('couldnt parse ' + text)
        return (np.nan,np.nan,np.nan,np.nan)
    
def parse_google_url(url):
    try:
        split = re.split('q=|%2C|\&zoom',url)
        return (float(split[1]),float(split[2]))
    except:
        print('couldnt parse ' + text)
        return (np.nan,np.nan)

def parse_landwatch(list_df,google_df):

	assert len(google_df) ==len(list_df)
	#google_df = google_df.drop('level_0',axis=1)

	df = pd.concat([list_df,google_df],axis=1)
	df = df[df['google_url']!='error']

	df['acres'], df['city'], df['county'], df['price'] = list(zip(*df['text'].apply(parse_text)))
	df['latitude'],df['longitude'] = list(zip(*df['google_url'].apply(parse_google_url)))
	parsed_df = df[['acres','price','latitude','longitude']].dropna()
	parsed_df['ppa'] = parsed_df['price']/parsed_df['acres']
	#parsed_df = parsed_df[parsed_df['longitude'] < -100] # remove incorrectly labeled points

	return parsed_df

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
#plot price per acre data in a state
def plot_ppa(parsed_df,state_code,ax = None):
	if ax is None:
		fig, ax = plt.subplots(figsize=(8,8))
	state_gdf = maps.get_state_gdf(state_code)
	#plot california
	base = state_gdf.plot(color='none',edgecolor='black',ax=ax)
	im = plt.scatter(parsed_df['longitude'],parsed_df['latitude'],c=parsed_df['ppa'],norm=LogNorm(),cmap='viridis')
	plt.axis('off')
	ax.set_aspect(aspect=1)

	#add custom colorbar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = fig.colorbar(im, cax=cax,ticks=[1e3,1e4,1e5])
	cbar.ax.set_yticklabels(['$1000','$10,000','$100,000'])

	ax.set_title('Price per acre',fontsize=16)
	return ax
	#plt.show()
