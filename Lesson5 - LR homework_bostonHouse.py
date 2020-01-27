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

#%%
"""
CRIM: 城镇人均犯罪率
ZN: 住宅用地所占比例
INDUS: 城镇中非住宅用地所占比例
CHAS: CHAS 虚拟变量,用于回归分析
NOX: 环保指数
RM: 每栋住宅的房间数
AGE: 1940 年以前建成的自住单位的比例
DIS: 距离 5 个波士顿的就业中心的加权距离。
RAD: 距离高速公路的便利指数
TAX: 每一万美元的不动产税率
PRTATIO: 城镇中的教师学生比例
B: 城镇中的黑人比例
LSTAT: 地区中有多少房东属于低收入人群
MEDV: 自住房屋房价中位数（也就是均价）
"""
boston = datasets.load_boston()
boston.keys()
x = boston.data
target = boston.target
print(x.shape)
print(target.shape)
print(boston.data.shape)
print(boston.feature_names)
print (boston.DESCR)


#%%
"""
1.训练数据:测试数据 = 2:8
2.调用LinearRegression 的fit方法
3.调用LinearRegression 的 predict方法
4.mean_squared_error,均方误差
5.mean_absolute_error,平均绝对误差
6.r2_score:拟合优度
"""
X_train,X_test,y_train,y_test = train_test_split(x,target,test_size=0.20)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

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
plt.ylabel("Predieted Price")
plt.xlabel("Real Price")

plt.show()

#%%
"""
参考代码：
https://www.kaggle.com/shreayan98c/boston-house-price-prediction

手动敲入代码和注释，并且对每个对应的函数进行解释，首次真正上手作业，用这样的笨办法来学习

"""
data = pd.DataFrame(x)
data.columns =  boston.feature_names
data.head(5)
#%%
"""
查看数据集的属性
"""
##adding target varible to dataframe
data['PRICE'] = boston.target
data.shape
data.columns
data.dtypes
## check the unique numbeer of values in dataset
data.nunique()
##check for missing values
data.isnull().sum()
##check the rows with missing values
data[data.isnull().any(axis =1)]
##viewin gthe data statistics
data.describe()
##Finding out the correlation between the features
corr = data.corr()
corr.shape
## plot the heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':15},cmap='Greens')
plt.show()

#%%
#split target varible and independent variables
## 将数据集中的PRICE列去除，y为输出结果
X = data.drop(['PRICE'],axis=1)
y = data['PRICE']
# Splitting to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)



#%%
##Linear regression
"""
Train the model
"""

#Train the model using training sets
lm = LinearRegression()
lm.fit(X_train,y_train)

#%%
"""
coef_:各特征的系数（重要性）
intercept_:截距的大小（常数值）
"""
lm.intercept_
## 将coefficient value 转换为dafaframe
coeffcients = pd.DataFrame([X_train.columns,lm.coef_]).T
coeffcients = coeffcients.rename(columns = {0: 'Attribute',1:'Coefficients'})
coeffcients


#%%
#Model prediction on train data
"""
𝑅^2 : It is a measure of the linear relationship between X and Y. It is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable.

Adjusted 𝑅^2 :The adjusted R-squared compares the explanatory power of regression models that contain different numbers of predictors.

MAE : It is the mean of the absolute value of the errors. It measures the difference between two continuous variables, here actual and predicted values of y. 

MSE: The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value. 

RMSE: The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value. 
"""

y_pred  = lm.predict(X_train)
#Model Evaulation
print('R^2:',metrics.r2_score(y_train,y_pred))
print('Adjusted R^2:' ,1- (1-metrics.r2_score(y_train,y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train,y_pred))
print('MSE:',metrics.mean_squared_error(y_train,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train,y_pred)))

#%%
plt.scatter(y_train,y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted Prices")
plt.show()

#%%
#check the residuals

plt.scatter(y_pred,y_train - y_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

#%%
#check the nomality of errors
sns.distplot(y_train - y_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

#%%
## predicting test data with model
y_test_pred = lm.predict(X_test)

#model Evaluation
acc_linreg = metrics.r2_score(y_test,y_test_pred)
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))