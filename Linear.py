# author: Chuanjie Guo
# contact: chuanjieguo@139.com

import pandas as pd
from sklearn import datasets, linear_model
iris = datasets.load_iris()
clf = linear_model.LinearRegression()
train_x = pd.DataFrame(iris.data).loc[:130]
train_y = pd.DataFrame(iris.target).loc[:130]

test_x = pd.DataFrame(iris.data).loc[130:]
test_y = pd.DataFrame(iris.target).loc[130:].reset_index(drop=True)

clf = clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
y_pred = pd.DataFrame(y_pred)

print("rate:%.2f" % ((test_y==y_pred).sum()/test_y.count()))