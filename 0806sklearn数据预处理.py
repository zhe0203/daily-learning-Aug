# -*- coding: utf-8 -*-
# 标准化、去均值、方差缩放
from sklearn import preprocessing
import numpy as np
X = np.array([[1,-1,2],[2,0,0],[0,1,-1]],dtype='float') # 设置数据类型dtype
X_scale = preprocessing.scale(X)
print('原数据为:\n',X)
print('标准化的数据为:\n',X_scale)
# 经过缩放的数据集具有零均值和标准差
print('均值为:\n',X_scale.mean(axis=0))
print('方差为:\n',X_scale.std(axis=0))

'''
在机器学习中，一般需要将数据集分为训练集和测试集进行训练模型，而在对于训练集进行数据预处理的时候，
需要将同样的方法运用在测试数据集上，这样才能保证数据标准化的方法一致
preprocessing模块提供了一个实用类，StandardScaler使用Transformer接口在训练集上计算均值和标准差，以便于在后续的测试集上进行相同的缩放
也可以通过在构造函数:class:StandardScaler`中传入参数``with_mean=False` 或者``with_std=False``来取消中心化或缩放操作。
'''
scaler = preprocessing.StandardScaler().fit(X)
print('均值为:\n',scaler.mean_)
print('标准差为:\n',scaler.scale_)
print(scaler.transform(X))   # 将相同的方法运用在其他数据集上

# 缩放类对象可以在新的数据集上实现和训练集相同缩放操作,使用transform函数
print('在新的数据集上的结果:\n',scaler.transform([[-1,1,0]]))

# 特征缩放至特定范围
'''
将特征缩放至给定的最小、最大值范围，经常是[0,1]
MinMaxScaler    MaxAbsScaler
'''
X_train = np.array([[1,-1,2],
                    [2,0,0],
                    [0,1,-1]],dtype='float')
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print('\n',X_train_minmax)
# 同样的转换实例可以被用在与在训练过程中不可见的测试数据
# 实现和训练数据一致的缩放和移位操作
X_test = np.array([[-3,-1,4]],dtype='float')
X_test_minmax = min_max_scaler.transform(X_test)
print('\n',X_test_minmax)

# 通过查看缩放器观察训练集中学习到的转换操作
print('\n',min_max_scaler.scale_,'\n',min_max_scaler.min_)

'''
X_std = (X-X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std / (max - min) + min
'''

'''
MaxAbsScaler 工作原理非常相似,但是它只通过除以每个特征的最大值将训练数据特征缩放至 [-1, 1]
'''
X_train = np.array([[1,-1,2],[2,0,0],[0,1,-1]],dtype='float')
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print(X_train_maxabs)
X_test = np.array([[-3,-1,4]],dtype='float')
X_test_maxabs = max_abs_scaler.transform(X_test)
print(X_test_maxabs)
print(max_abs_scaler.scale_)
'''
如果不想创造一个对象，可以使用scale minmax_scale maxabs_scale来现实快速缩放操作
'''

# 稀疏数据缩放
## 含异常值数据缩放
## 如果数据包含较多的异常值，使用均值和方差缩放并不是一个很好的选择，在这种情况下使用robust_scale使用更加鲁棒的中心和范围估计来缩放数据
df = np.array([[1,-100,2],[20,0,0],[0,1,-1]],dtype='float')
result_1 = preprocessing.robust_scale(df)
result_2 = preprocessing.MaxAbsScaler()
print('含有异常值:\n',result_1)
print('不含有异常值:\n',result_2.fit_transform(df))


# 规范化数据 normalize
X = [[1,-1,2],[2,0,0],[0,1,-1]]
x_normalized = preprocessing.normalize(X,norm='l2')
print(x_normalized)
'''
preprocessing``模块也提供了实用类 :class:`Normalizer` ，通过使用接口 ``Transformer 来实现相同的操作。
(在这里``fit``方法并没有作用: 因为规范化类在面对不同的样本数据时是无状态独立的)。
'''
normalizer = preprocessing.Normalizer().fit(X)
normalizer.transform(X)
normalizer.transform([[-1,1,0]])

# 二值化
'''
特征二值化是将数值型特征转变成布尔型特征
通过设置一定的阀值，将大于阀值的数值转变为1，将小于等于阀值的数值转变为0
现实数据的二值转换
模块也提供了二值化方法:func:`binarize，以便不需要转换接口时使用。
'''
X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]
binarizer = preprocessing.Binarizer().fit(X)
binarizer.transform(X)
# 可以改变二值器的阀值
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)

# 分类特征编码
'''
特征更多的时候是分类特征，而不是连续的数值特征。
比如一个人的特征可以是``[“male”, “female”]``， ["from Europe", "from US", "from Asia"]，
["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]。 这样的特征可以高效的编码成整数，
例如 ["male", "from US", "uses Internet Explorer"]``可以表示成 ``[0, 1, 3]，
["female", "from Asia", "uses Chrome"]``就是``[1, 2, 1]
'''
# 一个将分类器特征转换成scikit-learn估计器可用特征的方法是使用one-of-K
# 或者one-hot编码，该方法是OneHotEncoder实现，将每个类别的特征m可能值转换为,个二进制特征
enc = preprocessing.OneHotEncoder()
enc.fit([[0,0,3],[1,1,0],[0,2,1],[1,0,2]])
print(enc.transform([[0,1,3]]).toarray())

# 标签编码
le = preprocessing.LabelEncoder()
le.fit([1,2,2,6])
print(le.transform([1,1,2,6]))
# 非数值型转换为数值型
le.fit(['paris','tokyo','amsterdam'])
print(le.transform(['tokyo','paris'])) #　2 1

# 缺失值处理
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit([[1,2],[np.nan,3],[7,6]])
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

import scipy.sparse as sp
X = sp.csc_matrix([[1,2],[0,3],[7,6]])
imp = Imputer(missing_values=0,strategy='mean',axis=0)
imp.fit(X)
X_test = sp.csc_matrix([[0,2],[6,0],[7,6]])
print(imp.transform(X_test))

'''
很多情况下，考虑输入数据中的非线性特征来增加模型的复杂性是非常有效的。一个简单常用的方法就是使用多项式特征，它能捕捉到特征中高阶和相互作用的项。
'''
from sklearn.preprocessing import PolynomialFeatures
x = np.arange(6).reshape(3,2)
poly = PolynomialFeatures(2)
print(poly.fit_transform(x))
# 特征向量X从:math:(X_1, X_2) 被转换成:math:(1, X_1, X_2, X_1^2, X_1X_2, X_2^2)。
#　在一些情况中,我们只需要特征中的相互作用项 interaction_only=True
x = np.arange(9).reshape(3,3)
poly = PolynomialFeatures(degree=3,interaction_only=True)
print(poly.fit_transform(x))

# 装换器定制
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
x = np.array([[0,1],[2,3]])
transformer.transform(x)
