#%%
from sklearn import datasets
from sklearn.linear_model import LinearRegression
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
"""
Sex		nominal			M, F, and I (infant)
	Length		continuous	mm	Longest shell measurement
	Diameter	continuous	mm	perpendicular to length
	Height		continuous	mm	with meat in shell
	Whole weight	continuous	grams	whole abalone
	Shucked weight	continuous	grams	weight of meat
	Viscera weight	continuous	grams	gut weight (after bleeding)
	Shell weight	continuous	grams	after being dried
	Rings		integer			+1.5 gives the age in years
sns.pairplot(abalone,height = 2.5)
plt.tight_layout()
plt.show()
"""
abalone = pd.read_csv('/Users/rdi/Documents/ML/Bookola/data/abalone/abalone.data')
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight','Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
#%%
## check the unique numbeer of values in dataset
abalone.nunique()
##check for missing values
abalone.isnull().sum()
##check the rows with missing values
abalone[abalone.isnull().any(axis =1)]
##viewin gthe data statistics
abalone.describe()
##Finding out the correlation between the features
corr = abalone.corr()
## plot the heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':15},cmap='Greens')
plt.show()



#%%
###查看数据集的属性
abalone.head()
#%%
abalone.columns
#%%
abalone.dtypes
#describe() 查看每列数据的基本统计值，包括计数值、均值、标准差、最小最大值、1/4、1/2、3/4分位数
abalone.describe()
#info() 查看索引、数据类型和内存信息
abalone.info()
#%%
abalone.shape
#%%
"""
pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
data : array-like, Series, or DataFrame
输入的数据
prefix : string, list of strings, or dict of strings, default None
get_dummies转换后，列名的前缀
columns : list-like, default None
指定需要实现类别转换的列名
dummy_na : bool, default False
增加一列表示空缺值，如果False就忽略空缺值
drop_first : bool, default False
获得k中的k-1个类别值，去除第一个
"""

## Sex 列是M/F，需要替换为1，0 方便统计
##abalone.drop(columns='Sex')
sex = pd.get_dummies(abalone['Sex'])
abalone = pd.concat([abalone,sex],axis=1)
abalone.drop('Sex',axis=1,inplace = True)

#%%
X = abalone.drop(columns='Rings',axis=1)
y = abalone['Rings']
# Splitting to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)



#%%
##Linear regression
"""
Train the model
"""

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#%%
print('线性回归算法w值：', model.coef_)
print('线性回归算法b值: ', model.intercept_)
print("均方误差: %f" % mean_squared_error(y_test,y_pred))
print("平均绝对误差: %f" % mean_absolute_error(y_test,y_pred))
print("拟合优度: %f" % r2_score(y_test,y_pred))
#print("交叉验证: %f" % cross_val_score(y_test,y_pred))



#%%
"""
绘图

"""
plt.plot(y_pred,y_test,'rx')
plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'b-.', lw=4) # f(x)=x
plt.ylabel("Predieted Age")
plt.xlabel("Real Age")

plt.show()
