#%%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

#%%
data = pd.read_csv('/Users/rdi/Documents/ML/Bookola/data/master.csv', encoding='ISO-8859-1')

data.head(10)
data.describe()

#%%
##国家
uniqueCountry = data['country'].unique()
print(uniqueCountry)

#%%
##查看国家的数据的数量
plt.figure(figsize=(10, 25))
sns.countplot(y='country', data=data, alpha=0.6)
plt.title('按国家查看数据')
plt.show()

#%%
##查看性别
plt.figure(figsize=(16, 7))
sex = sns.countplot(x='sex', data=data)
plt.title('查看性别分布')
plt.show()
##基本一半一半

#%%
##使用热力图查看数据间的关联关系
##看起来GDP 与自杀率关联关系比较高
plt.figure(figsize=(16, 7))
cor = sns.heatmap(data.corr(), annot=True)
plt.title('各参数间的关联关系热力图')
plt.show()

#%%
##查看各年龄段的自杀者数量
##男性 35-54 岁的占比最高
plt.figure(figsize=(16, 7))
bar_age = sns.barplot(x='sex', y='suicides_no', hue='age', data=data)
plt.title('各年龄段自杀者数量')
plt.show()

#%%
##查看各代的自杀者数量
## boomer世代的自杀者数量较多，婴儿潮 - 背景知识了解
plt.figure(figsize=(16, 7))
bar_gen = sns.barplot(x='sex', y='suicides_no', hue='generation', data=data)
plt.title('各代自杀者数量')
plt.show()

# %%
##查看不同年份的不同年龄段的男女自杀状况
catAccordyear = sns.catplot(x='sex',y = 'suicides_no',hue = 'age',col = 'year',data = data,kind = 'bar',col_wrap=3)
plt.show()


#%%
## 查看不同年龄段的自杀数量

age_5 = data.loc[data.loc[:,'age']=='5-14 years', :]
age_15 = data.loc[data.loc[:,'age']=='15-24 years', :]
age_25 = data.loc[data.loc[:,'age']=='25-34 years', :]
age_35 = data.loc[data.loc[:,'age']=='35-54 years', :]
age_55 = data.loc[data.loc[:,'age']=='55-74 years', :]
age_75 = data.loc[data.loc[:,'age']=='75+ years', :]

plt.figure(figsize=(16,7))
age5Lp = sns.lineplot(x = 'year',y='suicides_no',data=age_5)
age15Lp = sns.lineplot(x = 'year',y='suicides_no',data=age_15)
age25Lp = sns.lineplot(x = 'year',y='suicides_no',data=age_25)
age35Lp = sns.lineplot(x = 'year',y='suicides_no',data=age_35)
age55Lp = sns.lineplot(x = 'year',y='suicides_no',data=age_55)
age75Lp = sns.lineplot(x = 'year',y='suicides_no',data=age_75)
leg = plt.legend(['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])

plt.show()

#%%
## 查看不同性别不同年度的自杀数量

malePopulation = data.loc[data.loc[:,'sex']=='male', :]
femalePopulation = data.loc[data.loc[:,'sex']=='female', :]

plt.figure(figsize=(16,7))
lpMalepopulation = sns.lineplot(x = 'year',y='suicides_no',data=malePopulation)
lpFemalepopulation = sns.lineplot(x = 'year',y='suicides_no',data=femalePopulation)
leg = plt.legend(['男', '女'])
##PRINT
plt.show()
