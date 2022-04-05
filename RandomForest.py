# author: Chuanjie Guo
# contact: chuanjieguo@139.com

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()

'''
["BaseEnsemble",
           "RandomForestClassifier", "RandomForestRegressor",
           "RandomTreesEmbedding", "ExtraTreesClassifier",
           "ExtraTreesRegressor", "BaggingClassifier",
           "BaggingRegressor", "GradientBoostingClassifier",
           "GradientBoostingRegressor", "AdaBoostClassifier",
           "AdaBoostRegressor", "VotingClassifier",
           "bagging", "forest", "gradient_boosting",
           "partial_dependence", "weight_boosting"]
'''
clf = RandomForestClassifier()
train_x = pd.DataFrame(iris.data).loc[:130]
train_y = pd.DataFrame(iris.target).loc[:130]

test_x = pd.DataFrame(iris.data).loc[130:]
test_y = pd.DataFrame(iris.target).loc[130:].reset_index(drop=True)

clf = clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
y_pred = pd.DataFrame(y_pred)

print("rate:%.2f" % ((test_y==y_pred).sum()/test_y.count()))