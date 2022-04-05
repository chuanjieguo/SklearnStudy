# author: Chuanjie Guo
# contact: chuanjieguo@139.com

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('iris_data.csv')
d1 = data[:]
d2 = data[:100]
d3 = data[100:]
d4 = data.loc[:, 'len1']

#print(d1)
#print(d2)
#print(d3)
print(d4)

ss = StandardScaler()
d5 = ss.fit_transform(data)
print(d5)


data = np.random.randn(10, 4)
scaler = StandardScaler()
scaler.fit(data)
trans_data = scaler.transform(data)
print('original data: ')
print(data)
print('transformed data: ')
print(trans_data)
print('scaler info: scaler.mean_: {}, scaler.var_: {}'.format(scaler.mean_, scaler.var_))
print('\n')