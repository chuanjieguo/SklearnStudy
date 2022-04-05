# author: Chuanjie Guo
# contact: chuanjieguo@139.com


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

X = pd.read_csv('hack_data.csv')
X.drop(['Location'], axis=1, inplace=True)

for i in range(2, 100):
    kmeans = KMeans(n_clusters=i)
    y_pred = kmeans.fit_predict(X)
    print("%d分类的分数：%.2f" % (i, metrics.calinski_harabaz_score(X, y_pred)))
