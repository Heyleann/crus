import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_curve,roc_curve, average_precision_score, precision_score, recall_score, roc_auc_score, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from singleClassifier import singleClassifier
from CRUS import CRUS

df = pd.read_csv('creditcard.csv')
df.dropna(how = 'any',inplace=True)
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
X = pd.get_dummies(X)

XTrain,XTest,yTrain,yTest = train_test_split(X,y,shuffle=True,test_size=0.1,stratify=y.values)

lgbmSg = singleClassifier(LGBMClassifier(n_jobs=8), XTrain, yTrain, XTest, yTest)
lgbmSg.sample()
params = {'boosting_type': ["gbdt", 'dart', 'goss'],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 6, 7, 8, 9, 10]}
lgbmSg.search(params)
lgbmSg.fit()
lgbmSg.predict()
lgbmSg.evaluate()
lgbmSg.plot()


features  = X.columns
X, y = X.values,y.values
lgbm = CRUS(LGBMClassifier(n_jobs=8),XTrain,yTrain,XTest,yTest,features)
lgbm.sample()
params = {'boosting_type':["gbdt",'dart','goss'],
              'learning_rate':[0.01,0.02,0.05,0.1],
              'n_estimators':[100,200,500],
              'max_depth':[5,6,7,8,9,10]}
lgbm.search(params=params)
clfs = [LGBMClassifier(n_jobs=8,**lgbm.optimalParams) for _ in range(len(lgbm.dataLoader))]
lgbm.fit(clfs)
lgbm.initializeWeights()
lgbm.predict()
lgbm.evaluate()
lgbm.plot()
lgbm.optimizeWeights('linear')
lgbm.predict()
lgbm.evaluate()
lgbm.plot()
