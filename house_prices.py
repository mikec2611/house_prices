import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')


def visualize_data(df_data):
	print('columns', df_data.columns)
	print('shape', df_data.shape)
	print('info', df_data.info())
	print('describe', df_data.describe())

	plt.figure(figsize = (30, 30))
	sns.heatmap(df_data.corr(), cmap = 'Blues', square = True, annot = True)
	plt.title("Correlations", size = 30)
	plt.show()


def process_data(df_data):
	# to-do -- pick only correlated fields

	# --- numeric fields
	df_numeric = df_data.select_dtypes(include=[np.number]) 
	
	# fill nan with mean
	df_fillmean = df_numeric[['LotFrontage', 'MasVnrArea']]
	df_fillmean.fillna(df_fillmean.mean(), inplace=True)
	df_numeric[['LotFrontage', 'MasVnrArea']] = df_fillmean[['LotFrontage', 'MasVnrArea']]

	# fill nan with mode -- year field
	df_fillmode = df_numeric['GarageYrBlt']
	df_fillmode.fillna(df_fillmode.mode()[0], inplace=True)
	df_numeric['GarageYrBlt'] = df_fillmode

	# add Id field
	df_numeric['Id'] = df_data['Id']

	# --- categorical fields
	df_category = df_data.select_dtypes(['object'])

	# remove bad char
	df_category.fillna('', inplace=True)
	df_category = df_category.applymap(lambda x:re.sub('\[.*?\]', '', x) ) # remove brackets
	df_category = df_category.applymap(lambda x:re.sub('<.*?>+', '', x) )  # remove <>
	df_category = df_category.applymap(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) ) # remove punctuation
	df_category = df_category.applymap(lambda x:re.sub('\n' , '', x) ) # remove newline

	# encode fields
	label_encoder = LabelEncoder()
	df_category = df_category.apply(label_encoder.fit_transform)
	
	# add Id field
	df_category['Id'] = df_data['Id']

	# combine dfs back to single df
	df_data = pd.merge(df_numeric, df_category, on='Id')

	df_data.drop(['Id'], axis=1, inplace=True)
	return df_data



# get data
raw_data_train = pd.read_csv('train.csv')
raw_data_test = pd.read_csv('test.csv')

# visualize_data(raw_data_train)

data_train = process_data(raw_data_train)
data_test = process_data(raw_data_test)

X = data_train.drop(['SalePrice'], axis=1)
y = data_train['SalePrice']

# to-do scale data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)


# xgb = XGBRegressor()
# param_grid = {'n_estimators': list(range(500, 1000, 100)), 'learning_rate': [0.001, 0.01, 0.1]}
# grid_search_xgb = GridSearchCV(xgb, param_grid, cv = 5)
# grid_search_xgb.fit(X_train, y_train)

# # save model
# pickle.dump(grid_search_xgb, open('model', 'wb'))
loaded_model = pickle.load(open('model', 'rb'))

# predictions
predictions = loaded_model.predict(data_test)

# rmse_xgb_grid = np.sqrt(mean_squared_error(y_test, predictions))
# print("The Root Mean Squared Error is:", rmse_xgb_grid)
# print("XGB Score is:", grid_search_xgb.score(X_test, y_test) * 100, "%")


# save predictions
results = np.array(list(zip(raw_data_test.Id,predictions)))
results = pd.DataFrame(results, columns=['Id', 'SalePrice'])
results['Id'] = results['Id'].astype(int)
print(results.dtypes)
results.to_csv('results.csv', index = False)