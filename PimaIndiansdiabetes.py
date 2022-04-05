
# coding: utf-8

# 有趣的事，Python永远不会缺席！欢迎关注小婷儿的博客 http://www.cnblogs.com/xxtalhr/      <br/>
# <img src='./加我哦.png'>

# # PimaIndiansdiabetes.csv 数据集介绍
# 链接：https://pan.baidu.com/s/1PyP_r8BMnLLE-2fkKEPqKA <br/>
# 提取码：vztm <br/>
# 
# 1、该数据集最初来自国家糖尿病/消化/肾脏疾病研究所。数据集的目标是基于数据集中包含的某些诊断测量来诊断性的预测 患者是否患有糖尿病。<br/>
# 2、从较大的数据库中选择这些实例有几个约束条件。尤其是，这里的所有患者都是Pima印第安至少21岁的女性。<br/>
# 3、数据集由多个医学预测变量和一个目标变量组成Outcome。预测变量包括患者的怀孕次数、BMI、胰岛素水平、年龄等。<br/>
# 
# 4、数据集的内容是皮马人的医疗记录，以及过去5年内是否有糖尿病。所有的数据都是数字，问题是（是否有糖尿病是1或0），是二分类问题。数据的数量级不同，有8个属性，1个类别：<br/>
# 
# 【1】Pregnancies：怀孕次数 <br/>
# 【2】Glucose：葡萄糖 <br/>
# 【3】BloodPressure：血压 (mm Hg) <br/>
# 【4】SkinThickness：皮层厚度 (mm) <br/>
# 【5】Insulin：胰岛素 2小时血清胰岛素（mu U / ml <br/>
# 【6】BMI：体重指数 （体重/身高）^2 <br/>
# 【7】DiabetesPedigreeFunction：糖尿病谱系功能 <br/>
# 【8】Age：年龄 （岁） <br/>
# 【9】Outcome：类标变量 （0或1）<br/>
# 

# ## 当我们拿到一批原始的数据
# 
# 1、首先要明确有多少特征，哪些是连续的，哪些是类别的。<br/>
# 2、检查有没有缺失值，对确实的特征选择恰当方式进行弥补，使数据完整。<br/>
# 3、对连续的数值型特征进行标准化，使得均值为0，方差为1。<br/>
# 4、对类别型的特征进行one-hot编码。<br/>
# 5、将需要转换成类别型数据的连续型数据进行二值化。<br/>
# 6、为防止过拟合或者其他原因，选择是否要将数据进行正则化。<br/>
# 7、在对数据进行初探之后发现效果不佳，可以尝试使用多项式方法，寻找非线性的关系。<br/>
# 8、根据实际问题分析是否需要对特征进行相应的函数转换。<br/>
# 

# # 1、加载库

# In[1]:



import pandas as pd  # 数据科学计算工具
import numpy as np  # 数值计算工具
import matplotlib.pyplot as plt  # 可视化
import seaborn as sns  # matplotlib的高级API
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline # 在Notebook里面作图/嵌图')


# # 2、读取数据

# In[6]:



pima = pd.read_csv('PimaIndiansdiabetes.csv')
pima.head()
#pima.head()默认前5行，pima.tail()默认最后5行，查看Series或者DataFrame对象的小样本，当然我们也可以传递一个自定义数字


# In[7]:


pima.shape,pima.keys(),type(pima)


# In[8]:


pima.describe()
# panda的describe描述属性，展示了每一个字段的
#【count条目统计，mean平均值，std标准值，min最小值，25%，50%中位数，75%，max最大值】


# In[213]:


pima.groupby('Outcome').size()
#按照是否发病分组，并展示每组的大小


# # 3、Data Visualization - 数据可视化

# In[214]:



pima.hist(figsize=(16, 14));
#查看每个字段的数据分布；figsize的参数显示的是每个子图的长和宽
# 后面加个分号就不会出现下面的输出
# array([[<matplotlib.axes._subplots.AxesSubplot object at 0x00000235316A7C50>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x00000235319287B8>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x0000023531945E48>],
#        [<matplotlib.axes._subplots.AxesSubplot object at 0x0000023531977518>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x000002353199FBA8>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x000002353199FBE0>],
#        [<matplotlib.axes._subplots.AxesSubplot object at 0x0000023531EA8908>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x0000023531ED1F98>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x0000023531F03668>]],
#       dtype=object)


# In[216]:


sns.pairplot(pima, vars=pima.columns,hue = 'Outcome')

# 报错LinAlgError: singular matrix


# In[218]:


sns.pairplot(pima, vars=pima.columns[:-1], hue='Outcome')
plt.show()
# seaborn常用命令
#【1】set_style()是用来设置主题的，Seaborn有5个预设好的主题：darkgrid、whitegrid、dark、white、ticks，默认为darkgrid
#【2】set()通过设置参数可以用来设置背景，调色板等，更加常用
#【3】displot()为hist加强版
#【4】kdeplot()为密度曲线图
#【5】boxplot()为箱图
#【6】joinplot()联合分布图
#【7】heatmap()热点图
#【8】pairplot()多变量图，可以支持各种类型的变量分析，是特征分析很好用的工具
# data：必不可少的数据；hue：用一个特征来显示图像上的颜色，类似于打标签；vars:只留几个特征两两比较，否则使用data的全部变量；


# In[ ]:


# sns.pairplot(pima,diag_kind='hist', hue='Outcome')
sns.pairplot(pima, diag_kind='hist');


# In[219]:


pima.plot(kind='box', subplots=True, layout=(3,3), sharex=False,sharey=False, figsize=(16,14));

# 箱线图（Boxplot）也称箱须图（Box-whisker Plot），是利用数据中的五个统计量：最小值、第一四分位数、中位数、第三四分位数与最大值
# 来描述数据的一种方法，它也可以粗略地看出数据是否具有有对称性，分布的分散程度等信息，特别可以用于对几个样本的比较。
# 通过盒图，在分析数据的时候，盒图能够有效地帮助我们识别数据的特征：
#  直观地识别数据集中的异常值(查看离群点)。
#  判断数据集的数据离散程度和偏向(观察盒子的长度，上下隔间的形状，以及胡须的长度)。

#pandas.plot作图：数据分为Series 和 DataFrame两种类型；现释义数据为DataFrame的参数

#【0】data:DataFrame
#【1】x:label or position,default None 指数据框列的标签或位置参数
#【2】y:label or position,default None 指数据框列的标签或位置参数
#【3】kind:str（line折线图、bar条形图、barh横向条形图、hist柱状图、
#               box箱线图、kde Kernel的密度估计图，主要对柱状图添加Kernel概率密度线、
#               density same as “kde”、area区域图、pie饼图、scatter散点图、hexbin）
#【4】subplots:boolean，default False，为每一列单独画一个子图
#【5】sharex:boolean，default True if ax is None else False
#【6】sharey:boolean,default False
#【7】loglog:boolean,default False,x轴/y轴同时使用log刻度


# In[220]:


pima.plot(kind='box', subplots=True, layout=(3,3), sharex=False,sharey=False, figsize=(16,14))


#pandas.plot作图：数据分为Series 和 DataFrame两种类型；现释义数据为DataFrame的参数

#【0】data:DataFrame
#【1】x:label or position,default None 指数据框列的标签或位置参数
#【2】y:label or position,default None 指数据框列的标签或位置参数
#【3】kind:str（line折线图、bar条形图、barh横向条形图、hist柱状图、
#               box箱线图、kde Kernel的密度估计图，主要对柱状图添加Kernel概率密度线、
#               density same as “kde”、area区域图、pie饼图、scatter散点图、hexbin）
#【4】subplots:boolean，default False，为每一列单独画一个子图
#【5】sharex:boolean，default True if ax is None else False
#【6】sharey:boolean,default False
#【7】loglog:boolean,default False,x轴/y轴同时使用log刻度


# In[221]:


c = pima.iloc[:,0:8].corr()# 选择特征列，去掉目标列
c


# In[222]:


corr = pima.corr()  # 计算变量的相关系数，得到一个N * N的矩阵
corr


# In[223]:


plt.subplots(figsize=(14,12)) # 可以先试用plt设置画布的大小，然后在作图，修改
sns.heatmap(corr, annot = True) # 使用热度图可视化这个相关系数矩阵


# 其生成的原理简单概括为四个步骤：

# （1）为离散点设定一个半径，创建一个缓冲区；

# （2）对每个离散点的缓冲区，使用渐进的灰度带（完整的灰度带是0~255）从内而外，由浅至深地填充；

# （3）由于灰度值可以叠加（值越大颜色越亮，在灰度带中则显得越白。在实际中，可以选择ARGB模型中任一通道作为叠加灰度值），
#     从而对于有缓冲区交叉的区域，可以叠加灰度值，因而缓冲区交叉的越多，灰度值越大，这块区域也就越“热”；

# （4）以叠加后的灰度值为索引，从一条有256种颜色的色带中（例如彩虹色）映射颜色，并对图像重新着色，从而实现热点图。


# # 4、Feature Extraction 特征提取

# In[224]:


# 导入和特征选择相关的包
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# SelectKBest() 只保留K个最高分的特征
# SelectPercentile() 只保留用户指定百分比的最高得分的特征
# 使用常见的单变量统计检验：假正率SelectFpr，错误发现率SelectFdr，或者总体错误率SelectFwe
# GenericUnivariateSelect通过结构化策略进行特征选择，通过超参数搜索估计器进行特征选择

# SelectKBest()和SelectPercentile()能够返回特征评价的得分和P值
#
# sklearn.feature_selection.SelectPercentile(score_func=<function f_classif>, percentile=10)
# sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, k=10)

# 其中的参数score_func有以下选项：

#【1】回归：f_regression:相关系数，计算每个变量与目标变量的相关系数，然后计算出F值和P值
#          mutual_info_regression:互信息，互信息度量X和Y共享的信息：
#         它度量知道这两个变量其中一个，对另一个不确定度减少的程度。
#【2】分类：chi2：卡方检验
#          f_classif:方差分析，计算方差分析（ANOVA）的F值（组间均方/组内均方）；
#          mutual_info_classif:互信息，互信息方法可以捕捉任何一种统计依赖，但是作为非参数方法，
#                              需要更多的样本进行准确的估计。


# In[225]:


X = pima.iloc[:, 0:8]  # 特征列 0-7列，不含第8列
Y = pima.iloc[:, 8]  # 目标列为第8列

select_top_4 = SelectKBest(score_func=chi2, k=4)  # 通过卡方检验选择4个得分最高的特征


# In[226]:


X.shape,pima.shape


# In[227]:


X.head()


# In[228]:


fit = select_top_4.fit(X, Y)  # 获取特征信息和目标值信息
features = fit.transform(X)  # 特征转换


# In[229]:


features[0:5]
#新特征列的前5行
# 因此，表现最佳的特征是：Glucose-葡萄糖、Insulin-胰岛素、BMI指数、Age-年龄


# In[230]:


# 构造新特征DataFrame
X_features = pd.DataFrame(data = features, columns=['Glucose','Insulin','BMI','Age']) 


# In[231]:


X_features.head()


# # 5、Standardization - 标准化

# In[232]:


# 它将属性值更改为 均值为0，标准差为1 的 高斯分布.
# 当算法期望输入特征处于高斯分布时，它非常有用
from sklearn.preprocessing import StandardScaler

# StandardScaler
# 作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。 StandardScaler对每列分别标准化，
# 因为shape of data: [n_samples, n_features]
# 【注：】 并不是所有的标准化都能给estimator带来好处。


# In[233]:


rescaledX = StandardScaler().fit_transform(
    X_features)  # 通过sklearn的preprocessing数据预处理中StandardScaler特征缩放 标准化特征信息
X = pd.DataFrame(data=rescaledX, columns=X_features.columns)  # 构建新特征DataFrame
X.head()


# # 6、机器学习 - 构建二分类算法模型

# In[234]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 切分数据集为：特征训练集、特征测试集、目标训练集、目标测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=2019, test_size=0.2)


# In[235]:


X.shape, Y.shape, X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[236]:


models = []
models.append(("LR", LogisticRegression()))  #逻辑回归
models.append(("NB", GaussianNB()))  # 高斯朴素贝叶斯
models.append(("KNN", KNeighborsClassifier()))  #K近邻分类
models.append(("DT", DecisionTreeClassifier()))  #决策树分类
models.append(("SVM", SVC()))  # 支持向量机分类


# In[237]:


models


# In[238]:


import warnings
warnings.filterwarnings('ignore')  #消除警告

results = []
names = []
for name, model in models:
    kflod = KFold(n_splits=10, random_state=2019)
    cv_result = cross_val_score(
        model, X_train, Y_train, cv=kflod, scoring='accuracy')
    names.append(name)
    results.append(cv_result)
print(results,names)

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
for i in range(len(names)):
    print(names[i], results[i].mean)


# In[239]:


results,names


# # 7、 基于PCA和网格搜索SVM参数
# 
# 1、PCA是常用的提取数据的手段，其功能为提取主成分（主要信息），摒弃冗余信息（次要信息），从而得到压缩后的数据，实现维度的下降。<br/>
# 2、其设想通过投影矩阵将高维信息转换到另一个坐标系下，并通过平移将数据均值变为零。PCA认为，在变换过后的数据中，在某一维度上，数据分布的更分散，则认为对数据点分布情况的解释力就更强。故在PCA中，通过方差来衡量数据样本在各个方向上投影的分布情况，进而对有效的低维方向进行选择。<br/>
# 
# 3、KernelPCA是PCA的一个改进版，它将非线性可分的数据转换到一个适合对齐进行线性分类的新的低维子空间上，核PCA可以通过非线性映射将数据转换到一个高维空间中，在高维空间中使用PCA将其映射到另一个低维空间中，并通过线性分类器对样本进行划分。<br/>
# 
# 4、核函数：通过两个向量点积来度量向量间相似度的函数。常用函数有：多项式核、双曲正切核、径向基和函数（RBF）（高斯核函数）等。<br/>
# 
# 5、KPCA和PCA都是用来做无监督数据处理的，但是有一点不一样。PCA是降维，把m维的数据降至k维。KPCA恰恰相反，它是把m维的数据升至k维。但是他们共同的目标都是让数据在目标维度中（线性）可分，即PCA的最大可分性。在sklearn中，kpca和pca的使用基本一致，接口都是一样的。kpca需要指定核函数，不然默认线性核。

# In[240]:


# 【1】 Applying Kernel PCA

from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf')
X_train_pca = kpca.fit_transform(X_train)
X_test_pca = kpca.transform(X_test)

# fit和transform没有任何关系，仅仅是数据处理的两个不同环节，之所以出来fit_transform这个函数名，仅仅是为了写代码方便，会高效一点。

# sklearn里的封装好的各种算法使用前都要fit，fit相对于整个代码而言，为后续API服务。fit之后，然后调用各种API方法，
# transform只是其中一个API方法，所以当你调用transform之外的方法，也必须要先fit。

# fit原义指的是安装、使适合的意思，其实有点train的含义，但是和train不同的是，它并不是一个训练的过程，而是一个适配的过程，
# 过程都是确定的，最后得到一个可用于转换的有价值的信息。

# 1、必须先用fit_transform(trainData)，之后再transform(testData)
# 2、如果直接transform(testData)，程序会报错
# 3、如果fit_transfrom(trainData)后，使用fit_transform(testData)而不transform(testData)，虽然也能归一化，
# 但是两个结果不是在同一个“标准”下的，具有明显差异。(一定要避免这种情况)

# 4、fit_transform()干了两件事：fit找到数据转换规则，并将数据标准化
# 5、transform()可以直接把转换规则拿来用，所以并不需要fit_transform()，否则，两次标准化后的数据格式就不一样了


# # 8、 SVM旨在将一组不可线性分割的数据线性分割。
# ##### 核函数：通过两个向量点积来度量向量间相似度的函数。常用函数有：多项式核、双曲正切核、径向基和函数（RBF）（高斯核函数）等。
# 这些函数中应用最广的应该就是RBF核了，无论是小样本还是大样本，高维还是低维等情况，RBF核函数均适用，它相比其他的函数有一下优点：<br/>
# 1）RBF核函数可以将一个样本映射到一个更高维的空间，而且线性核函数是RBF的一个特例，也就是说如果考虑使用RBF，那么就没有必要考虑线性核函数了。<br/>
# 2）与多项式核函数相比，RBF需要确定的参数要少，核函数参数的多少直接影响函数的复杂程度。另外，当多项式的阶数比较高时，核矩阵的元素值将趋于无穷大或无穷小，而RBF则在上，会减少数值的计算困难。<br/>
# 3）对于某些参数，RBF和sigmoid具有相似的性能。<br/>
# 

# In[241]:


X_train.shape,X_test.shape,X_train_pca.shape,X_test_pca.shape,Y_train.shape


# In[242]:


plt.figure(figsize=(10,8))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1],c=Y_train,cmap='plasma')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")


# In[246]:


plt.figure(figsize=(10,8))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1],c=Y_train)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")


# # 9、SVC

# In[247]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

classifier = SVC(kernel = 'rbf')
classifier.fit(X_train_pca, Y_train)


# # 10、classification_report简介
# sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。 
# 主要参数: <br/>
# y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。  <br/>
# y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。  <br/>
# labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。  <br/>
# target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。  <br/>
# sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。  <br/>
# digits：int，输出浮点值的位数．

# In[250]:


# 使用SVC预测生存

y_pred = classifier.predict(X_test_pca)
cm = confusion_matrix(Y_test, y_pred)
print(cm)#预测真确和错误的个数
print('++++++++++++++++++++++++++++++++')
print(classification_report(Y_test, y_pred))

# 混淆矩阵（confusion_matrix）。

# TP(True Positive)：将正类预测为正类数，真实为0，预测也为0
# FN(False Negative)：将正类预测为负类数，真实为0，预测为1
# FP(False Positive)：将负类预测为正类数， 真实为1，预测为0
# TN(True Negative)：将负类预测为负类数，真实为1，预测也为1

# 精确率(precision)分母为预测为正样例的个数，分子为预测为实际正样例被预测准的个数
# 召回率(recall)分母为实际正样例的个数，分子为预测为实际正样例被预测准的个数
# F1-score混合的度量，对不平衡类别非常有效
# 准确率(accuracy)模型的整体的性能的评估
# Specificity分母为实际负样例的个数，分子为预测为实际负样例被预测准的个数
# 右边support列为每个标签的出现次数．avg / total行为各列的均值（support列为总和）． 


#  # 11、使用 网格搜索 来提高模型
# 1、GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。<br/>
#     但是这个方法适合于小数据集，一旦数据的量级上去了，很难得出结果<br/>
# 2、C: float参数 默认值为1.0<br/>
#     错误项的惩罚系数。C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率<br/>
#     降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，<br/>
#     把训练样本集中错误分类的样本作为噪声。<br/>
# 3、gamma：float参数 默认为auto<br/>
# 
#    核函数系数，只对‘rbf’,‘poly’,‘sigmod’有效。<br/>
# 
#    如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features.<br/>

# In[251]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001]};
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose = 2);
grid.fit(X_train_pca, Y_train);

# 预测
grid_predictions = grid.predict(X_test_pca);

# 分类报告
print(classification_report(Y_test,grid_predictions))


# # 12、可视化结果

# In[257]:


results


# In[256]:


ax = sns.boxplot(data = results)
ax.set_xticklabels(names)

# 通过盒图，在分析数据的时候，盒图能够有效地帮助我们识别数据的特征：
#  直观地识别数据集中的异常值(查看离群点)。
#  判断数据集的数据离散程度和偏向(观察盒子的长度，上下隔间的形状，以及胡须的长度)。


# # 13、中位数的求法
# 
# 2710 2755 2850 | 2880 2880 2890 | 2920 2940 2950 | 3050 3130 3325<br/>
# 
# Q1 = 2865　Q2 = 2905(中位数)　Q3 = 3000<br/>
# 
# 中位数是2 905，第一个四分位数Q1 = 2865，第三个四分位数Q3 = 3000。检查这些数据，最小值为2710，<br/>
# 最大值为3325。因此，薪水数据的五数概括数据为2710、2865、2905、3000、3325。大约1／4或25％的观察值在五数概括的相邻两个数字之间。<br/>
# 
# 
# a = [2755 ,2850,  2880 ,2880 ,2710, 2890, 2920 ,2940, 2950, 3050 ,3130, 3325]<br/>
# 
# 2905 = (2890+2920)/2     # 2*(12+1)/4=6.5 取6、7位的数的平均值<br/>
# 2865 = (2850+2880)/2     # (12+1)/4=3.25 取2、3位的数的平均值<br/>
# 3000 = (2950+3050)/2     # 3*(12+1)/4=9.75 取7、10位的数的平均值<br/>

# # 14、使用逻辑回归预测 

# In[262]:


lr = LogisticRegression() # LR模型构建
lr.fit(X_train, Y_train) # 
predictions = lr.predict(X_test) # 使用测试值预测


# In[263]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[265]:


print(accuracy_score(Y_test, predictions)) # 打印评估指标（分类准确率）
print(classification_report(Y_test,predictions)) 


# In[267]:


conf = confusion_matrix(Y_test, predictions) # 混淆矩阵
label = ["0","1"] # 
sns.heatmap(conf, annot = True, xticklabels=label, yticklabels=label)


# # 15、补充
# 数据预处理方法
# 1. 去除唯一属性<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;唯一属性通常是一些id属性，这些属性并不能刻画样本自身的分布规律，所以简单地删除这些属性即可。<br/>
# 2. 缺失值处理的三种方法：<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;直接使用含有缺失值的特征；<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;删除含有缺失值的特征（该方法在包含缺失值的属性含有大量缺失值而仅仅包含极少量有效值时是有效的）；<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;缺失值补全。<br/>
# 3.常见的缺失值补全方法：<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;（1）均值插补<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;如果样本属性的距离是可度量的，则使用该属性有效值的平均值来插补缺失的值；<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;如果的距离是不可度量的，则使用该属性有效值的众数来插补缺失的值。如果使用众数插补，出现数据倾斜会造成什么影响？<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;（2）同类均值插补<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;首先将样本进行分类，然后以该类中样本的均值来插补缺失值。<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;（3）建模预测<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;将缺失的属性作为预测目标来预测，将数据集按照是否含有特定属性的缺失值分为两类，利用现有的机器学习算法对待预测数据集的缺失值进行预测。<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;该方法的根本的缺陷是如果其他属性和缺失属性无关，则预测的结果毫无意义；但是若预测结果相当准确，则说明这个缺失属性是没必要纳入数据集中的；一般的情况是介于两者之间。<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;（4）高维映射<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;将属性映射到高维空间，采用独热码编码（one-hot）技术。将包含K个离散取值范围的属性值扩展为K+1个属性值，若该属性值缺失，则扩展后的第K+1个属性值置为1。这种做法是最精确的做法，保留了所有的信息，也未添加任何额外信息，若预处理时把所有的变量都这样处理，会大大增加数据的维度。这样做的好处是完整保留了原始数据的全部信息、不用考虑缺失值；缺点是计算量大大提升，且只有在样本量非常大的时候效果才好。<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;（5）多重插补（MultipleImputation，MI）<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;多重插补认为待插补的值是随机的，实践上通常是估计出待插补的值，再加上不同的噪声，形成多组可选插补值，根据某种选择依据，选取最合适的插补值。<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;（6）压缩感知和矩阵补全<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;（7）手动插补<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;插补处理只是将未知值补以我们的主观估计值，不一定完全符合客观事实。在许多情况下，根据对所在领域的理解，手动对缺失值进行插补的效果会更好。
