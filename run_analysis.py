# ============================================================
# CS301 - Milestone 2: Crime vs Housing Prices in NYC
# Team: Maksym Soroka, Anthony Chahla
# Docker entrypoint script — generates all visualizations
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for Docker — no display available
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

os.makedirs('/output', exist_ok=True)
sns.set_theme(style="whitegrid")
print("Libraries loaded ✅")

# --- Load Data (local CSVs copied into container) ---
crime_df = pd.read_csv('processed_nypd.csv')
housing_df = pd.read_csv('nyc_housing_base.csv')
print(f"Crime: {crime_df.shape} | Housing: {housing_df.shape} ✅")

# --- Clean & Merge ---
borough_map_crime = {'K': 'Brooklyn', 'Q': 'Queens', 'M': 'Manhattan',
                     'S': 'Staten Island', 'B': 'Bronx'}
crime_df['borough_name'] = crime_df['ARREST_BORO'].map(borough_map_crime)
crime_counts = crime_df.groupby('borough_name').size().reset_index(name='crime_count')

borough_map_housing = {1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn',
                       4: 'Queens', 5: 'Staten Island'}
housing_df['borough_name'] = housing_df['borough_x'].map(borough_map_housing)
housing_df = housing_df[housing_df['sale_price'] > 10000]

avg_price = housing_df.groupby('borough_name')['sale_price'].median().reset_index()
avg_price.columns = ['borough_name', 'median_sale_price']

merged_df = pd.merge(crime_counts, avg_price, on='borough_name')
print("Data merged ✅")

# -------------------------------------------------------
# PLOT 1: EDA Overview
# -------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('EDA: Crime vs Housing Prices in NYC', fontsize=16, fontweight='bold')

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

axes[0, 0].bar(merged_df['borough_name'], merged_df['crime_count'], color=colors)
axes[0, 0].set_title('Crime Count by Borough')
axes[0, 0].set_xlabel('Borough')
axes[0, 0].set_ylabel('Number of Arrests')
axes[0, 0].tick_params(axis='x', rotation=15)

axes[0, 1].bar(merged_df['borough_name'], merged_df['median_sale_price'], color=colors)
axes[0, 1].set_title('Median Sale Price by Borough')
axes[0, 1].set_xlabel('Borough')
axes[0, 1].set_ylabel('Median Price ($)')
axes[0, 1].tick_params(axis='x', rotation=15)

axes[1, 0].scatter(merged_df['crime_count'], merged_df['median_sale_price'],
                   color='steelblue', s=200, zorder=5)
for _, row in merged_df.iterrows():
    axes[1, 0].annotate(row['borough_name'],
                        (row['crime_count'], row['median_sale_price']),
                        textcoords='offset points', xytext=(8, 4), fontsize=9)
axes[1, 0].set_title('Crime Count vs Median Sale Price')
axes[1, 0].set_xlabel('Crime Count')
axes[1, 0].set_ylabel('Median Sale Price ($)')

axes[1, 1].hist(housing_df['sale_price'].clip(upper=5000000),
                bins=50, color='steelblue', edgecolor='white')
axes[1, 1].set_title('Distribution of Sale Prices')
axes[1, 1].set_xlabel('Sale Price ($)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('/output/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plots saved ✅")

# -------------------------------------------------------
# PLOT 2: Correlation Heatmap
# -------------------------------------------------------
feature_cols = ['sale_price', 'yearbuilt', 'lotarea', 'bldgarea',
                'unitsres', 'unitstotal', 'numfloors', 'building_age']
corr_data = housing_df[feature_cols].dropna()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix: Housing Features')
plt.tight_layout()
plt.savefig('/output/correlation_heatmap.png', dpi=150)
plt.close()
print("Correlation heatmap saved ✅")

# -------------------------------------------------------
# PLOT 3: Linear Regression
# -------------------------------------------------------
X = merged_df[['crime_count']]
y = merged_df['median_sale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred) ** 0.5
print(f"R²: {r2:.4f} | RMSE: ${rmse:,.0f}")

plt.figure(figsize=(8, 5))
plt.scatter(merged_df['crime_count'], merged_df['median_sale_price'],
            color='steelblue', s=200, zorder=5, label='Boroughs')
for _, row in merged_df.iterrows():
    plt.annotate(row['borough_name'],
                 (row['crime_count'], row['median_sale_price']),
                 textcoords='offset points', xytext=(8, 4), fontsize=9)
plt.plot(merged_df['crime_count'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: Crime Count vs Median Sale Price')
plt.xlabel('Crime Count')
plt.ylabel('Median Sale Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('/output/regression_plot.png', dpi=150)
plt.close()
print("Regression plot saved ✅")

# -------------------------------------------------------
# PLOT 4: Bivariate Box Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Bivariate Analysis: Borough vs Crime & Housing Price',
             fontsize=14, fontweight='bold')

borough_order = ['Manhattan', 'Brooklyn', 'Queens', 'Staten Island', 'Bronx']
housing_filtered = housing_df[housing_df['sale_price'] < 3000000]

sns.boxplot(data=housing_filtered, x='borough_name', y='sale_price',
            order=borough_order, palette='Set2', ax=axes[0])
axes[0].set_title('Sale Price Distribution by Borough')
axes[0].set_xlabel('Borough')
axes[0].set_ylabel('Sale Price ($)')
axes[0].tick_params(axis='x', rotation=15)

crime_counts_type = crime_df.groupby(['borough_name', 'LAW_CAT_CD']).size().reset_index(name='count')
sns.boxplot(data=crime_counts_type, x='borough_name', y='count',
            order=borough_order, palette='Set1', ax=axes[1])
axes[1].set_title('Crime Frequency Distribution by Borough')
axes[1].set_xlabel('Borough')
axes[1].set_ylabel('Number of Incidents')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('/output/boxplot_analysis.png', dpi=150)
plt.close()
print("Box plots saved ✅")

print("\n✅ All 4 visualizations saved to /output/")
