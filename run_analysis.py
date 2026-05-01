import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

os.makedirs('/output', exist_ok=True)

# Load data
crime_df = pd.read_csv('processed_nypd.csv')
housing_df = pd.read_csv('nyc_housing_base.csv')

# Clean and merge
borough_map_crime = {'K': 'Brooklyn', 'Q': 'Queens', 'M': 'Manhattan',
                     'S': 'Staten Island', 'B': 'Bronx'}
crime_df['borough_name'] = crime_df['ARREST_BORO'].map(borough_map_crime)

borough_map_housing = {1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn',
                       4: 'Queens', 5: 'Staten Island'}
housing_df['borough_name'] = housing_df['borough_x'].map(borough_map_housing)
housing_df = housing_df[housing_df['sale_price'] > 10000]
