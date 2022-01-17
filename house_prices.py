import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# -steps
# get data 
# split data
# describe data
# clean text
# tokenize text
# remove stopwords
# stem text
# vectorize text


# get data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print(data_train.head(100))