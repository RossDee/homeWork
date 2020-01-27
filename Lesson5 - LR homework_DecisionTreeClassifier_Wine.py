#%%
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

#%%
wine = datasets.load_wine()
wine.keys()
x = wine.data
target = wine.target
print(wine.feature_names)
print (wine.DESCR)
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
data = pd.DataFrame(x)
data.columns =  wine.feature_names
#%%
wine.feature_names
#%%
wine.target_names
#%%
data.head()

#%%
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.4)
Xtrain.shape
Xtest.shape

#%%
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest) #返回预测的准确度
#%%
score


#%%
