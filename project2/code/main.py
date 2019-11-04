# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import matplotlib.pyplot as plt
from gmm import *
from sklearn.decomposition import PCA


# 设置调试模式
DEBUG = False

# 载入数据
res = np.loadtxt("cho.txt")
Y = res[:, 3:]
y = res[:,1]
ind = res[:, 0]
matY = np.matrix(Y, copy=True)

# 模型个数，即聚类的类别个数
K = 5

# 计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, 100)

# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]
# 求当前模型参数下，各模型对样本的响应度矩阵
gamma = getExpectation(matY, mu, cov, alpha)
# 对每个样本，求响应度最大的模型下标，作为其类别标识
category = gamma.argmax(axis=1).flatten().tolist()[0]
# 将每个样本放入对应类别的列表中
c = []
for j in range(K):
    c.append(np.array([int(ind[i]) for i in range(N) if category[i] == (j)]))



# Generate predict y
predict = np.zeros(y.shape, dtype=int)
for i in range(len(c)):
    for j in c[i]:
        predict[j-1] = i+1


name = 'iyer'
# Calculate rand index and jaccard coefficient
TP = TN = FP = FN = 0
for i in range(len(y)):
    for j in range(i+1, len(y)):
        if (predict[i]==predict[j] and y[i]==y[j]):
            TP += 1
        if (predict[i]!=predict[j] and y[i]!=y[j]):
            TN += 1
        if (predict[i]==predict[j] and y[i]!=y[j]):
            FP += 1
        if (predict[i]!=predict[j] and y[i]==y[j]):
            FN += 1
print('rand index of',name,'is',(TP+TN)/(TP+TN+FP+FN))
print('jaccard coefficient of',name,'is',TP/(TP+FP+FN))

# Use PCA to reduce dimension
pca = PCA(n_components=2)
predict = pca.fit_transform(Y)
# print(predict)
plt.title("GMM of sample " + str(name) + "\nrand index "+str((TP + TN) / (TP + TN + FP + FN)) + "\njaccard corfficient " + str(TP / (TP + FP + FN)))
for i in range(len(c)):
    # print(i)
    plt.scatter(predict[c[i]-1, 0], predict[c[i]-1, 1], label='class' + str(i))
plt.legend(loc='best')
plt.show()


