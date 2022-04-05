# author: Chuanjie Guo
# contact: chuanjieguo@139.com

import pandas as pd
from sklearn import datasets
from sklearn.externals import joblib
import pickle
iris = datasets.load_iris()
from sklearn.naive_bayes import  GaussianNB
clf = GaussianNB() # type: sklearn.naive_bayes.GaussianNB
train_x = pd.DataFrame(iris.data).loc[:130]
train_y = pd.DataFrame(iris.target).loc[:130]

test_x = pd.DataFrame(iris.data).loc[130:]
test_y = pd.DataFrame(iris.target).loc[130:].reset_index(drop=True)

clf = clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
y_pred1 = clf.predict_proba(test_x)
y_pred2 = clf.predict_log_proba(test_x)

print(pd.DataFrame(y_pred1))
print(pd.DataFrame(y_pred2))

y_pred = pd.DataFrame(y_pred)

print(y_pred)

print("rate:%.2f" % ((test_y==y_pred).sum()/test_y.count()))


#python提供两种模型持久化的方法，第一种是joblib,一种是pickle:如下：
#第一种：比如gnb是我们想要保存的模型
joblib.dump(clf, 'GaussianNB.model') #模型保存到本地
#lr = joblib.load('GauaaianNB.model') #重新加载模型
"""
重新加载模型，  需要注意的是，这里执行joblib.dump()之后，有可能还会生成若干个以rf.model_XX.npy为命名格式的文件，
这有可能是用于保存模型中的系数等的二进制文件。
其具体生成的文件的个数还会随调用到的分类器的不同，以及分类器中迭代次数的参数的不同而变，有时候会生成几个，有时候会生成几百个。
"""

#或者第二种方式
save = pickle.dumps(clf) #模保存模型
#model = pickle.loads(save)#重新导入模型