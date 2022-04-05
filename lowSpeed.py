#!/usr/bin/env python
# coding: utf-8

# # 案例描述

# 造成LTE网络小区下载速率较低的原因主要是在无线侧，其次还有一些是传输问题或者是核心网问题所致，此外，还可以通过参数调整优化及一些特殊新功能的开启来进行下载速率的提升。但是，如果能提前预知低速率的小区，对于网优人员，可以更好的主动的解决网络问题，提升用户满意度。

# # 训练和测试数据

# In[64]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn import metrics


# In[65]:


df = pd.read_csv(r'csvData/lowspeed.csv',sep=',',encoding='utf-8')


# # 整理、准备、探索数据

# In[66]:


# 字段描述
# cgi		小区标识
# LOWSPEED		是否低速率小区
# SUC_CALL_RATE		无线接通率
# PRB_UTILIZE_RATE		无线利用率
# SUC_CALL_RATE_QCI1		volte无线接通率
# mr		MR覆盖率
# cover_type		覆盖类型
# erabnbrmaxestab1		QCI1最大E-RAB数
# upspeed		网络上行速率


# In[67]:


# 数据没有缺失，'cgi'字段有英文字符
df.info()


# In[68]:


#样本数据明显不均衡，正样本52905，负样本2774，需要做样本均衡处理。
df['LOWSPEED'].value_counts().to_dict()


# # 清洗与预处理数据

# In[69]:


# #去掉字符串类型的列
df.drop('cgi',axis=1,inplace=True)
df.head(2)


# In[70]:


'''随机选择2774条正样本数据与2774条负样本数据，合并为一个新的二维数组。'''
#索引--异常
fraud_indices = np.array(df[df['LOWSPEED']==1].index)

#索引--正常
normal_indices = np.array(df[df['LOWSPEED']==0].index)

#索引--正常--随机取len(fraud_indices)
randome_normal_indices = np.random.choice(normal_indices,len(fraud_indices),replace=False)

#索引合并
indices = np.concatenate([fraud_indices,randome_normal_indices])

#更具索引取数据
df = df.loc[indices]

df['LOWSPEED'].value_counts().to_dict()


# In[71]:


# 数据拆分为x,y即feature与label
cols = df.columns.values.tolist()
df1 = df.copy()
x = df1.loc[:,[col for col in cols if col!='LOWSPEED']]
print(x.shape)
y = df.loc[:,['LOWSPEED']]
print(y.shape)


# In[72]:


# 数据拆分为训练与测试，比例为0.7：0.3
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[73]:


# 数据标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)


# # 模型训练与预测

# In[74]:


from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(bootstrap=True,oob_score=True,criterion='gini')
m.fit(x_train_std,y_train)
y_pre = m.predict(x_test_std)


# # 模型评估

# In[75]:


#综合报告
print(metrics.classification_report(y_test,y_pre))


# In[76]:


#混合矩阵
metrics.confusion_matrix(y_test,y_pre)


# In[77]:


#准确率分数
metrics.accuracy_score(y_test,y_pre)


# In[ ]:




