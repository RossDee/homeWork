#%%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels as sm

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

#%%
"""
1.导入数据，需要注意编码方式 ISO-8859-1
2.显示数据前十行，观察数据
"""
dating = pd.read_csv('/Users/rdi/Documents/ML/Bookola/data/Speed Dating Data.csv',encoding='ISO-8859-1')

dating.head(10)

#%%
"""
清洗数据，查看数据的空值
"""
dating.isnull().sum()

#%%
"""
用到的numpy 的isfinite
np.isfinite(ndarray)返回一个判断是否是有穷（非inf，非NaN）的bool型数组
不同年龄段的约会频次，23-28岁间的用户比较多

"""

age = dating[np.isfinite(dating['age'])]['age']
plt.hist(age.values)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


#%%
"""
第一次快速约会找到对象的比例
pandas crosstab
 crosstab(index: Any,
             columns: Any,
             values: Any = None,
             rownames: Any = None,
             colnames: Any = None,
             aggfunc: Any = None,
             margins: Any = False,
             margins_name: Any = "All",
             dropna: Any = True,
             normalize: Any = False) -> Any
index – Values to group by in the rows.
columns – Values to group by in the columns.
values – Array of values to aggregate according to the factors. Requires `aggfunc` be specified.

"""
pd.crosstab(index = dating['match'],columns="count")

#%%
