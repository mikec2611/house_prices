import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import lightgbm as lgb


import warnings
warnings.filterwarnings('ignore')


# get data
raw_data_train = pd.read_csv('train.csv')
raw_data_test = pd.read_csv('test.csv')

data_train = raw_data_train.drop("Id", axis=1)
data_test = raw_data_test.drop("Id", axis=1)

data_train.fillna('', inplace=True)
data_test.fillna('', inplace=True)

# visualize data
# fig = plt.subplots()
# sns.distplot(data_train['SalePrice'],fit=norm)
# res = stats.probplot(data_train['SalePrice'],plot = plt,dist='norm')
# plt.show()

# normalize SalePrice

# split data
X_train, X_test, y_train, y_test = train_test_split(data_train, y_train, test_size=0.2, random_state=101)

train = xgb.DMatrix(X_train, label=y_train)

param = {
    'max_depth':4,
    'eta':0.3,
    'objective': 'multi:softmax',
    'num_class':3
}
epochs= 10

model = xgb.train(param, train, epochs)

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))

