# author: Chuanjie Guo
# contact: chuanjieguo@139.com

'''
https://blog.csdn.net/mingtian715/article/details/54015798
'''
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)  # 随机种子，用于初始化聚类中心

digits = load_digits()  # 载入数据
data = scale(digits.data)  # 中心标准化数据

n_samples, n_features = data.shape  # 1797，64
n_digits = len(np.unique(digits.target))  # 10
labels = digits.target  # 数据真实标签

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

print(79 * '_')
print('% 9s' % 'init'
               '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


# 算法评估函数，包括处理时间、最终代价函数值、以及不同的聚类评价指标
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


# Kmeans++，随机初始化10次
bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

# Kmeans，随机初始化10次
bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# Pca+kmeans，采用如下调用方式聚类中心是确定的，因此只初始化一次
pca = PCA(n_components=n_digits).fit(data)  # fit函数表示拟合数据
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(79 * '_')