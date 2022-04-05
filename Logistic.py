# author: Chuanjie Guo
# contact: chuanjieguo@139.com

import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
clf = LogisticRegression()
train_x = pd.DataFrame(iris.data).loc[:130]
train_y = pd.DataFrame(iris.target).loc[:130]

test_x = pd.DataFrame(iris.data).loc[130:]
test_y = pd.DataFrame(iris.target).loc[130:].reset_index(drop=True)

clf = clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
y_pred = pd.DataFrame(y_pred)
y_pred1 = clf.predict_proba(test_x)
y_pred2 = clf.predict_log_proba(test_x)

print(pd.DataFrame(y_pred1))
print(pd.DataFrame(y_pred2))
print(y_pred)

print("rate:%.2f" % ((test_y==y_pred).sum()/test_y.count()))