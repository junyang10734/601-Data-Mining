import sys

sys.path.append("..")

from utils.similarity import calEuclidDistanceMatrix
from utils.knn import myKNN
from utils.laplacian import calLaplacianMatrix
from utils.dataloader import genTwoCircles
from utils.ploter import plot
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle, islice
from sklearn.decomposition import PCA

# np.random.seed(1)
def loadDataSet(filename):
    with open('./data/' + filename + '.txt', 'r') as f:
        data = f.readlines()
        dataList = []
        for i in range(len(data)):
            item = data[i].split('\t')
            item[-1] = item[-1].strip('\n')
            newitem = list(map(lambda x: float(x), item))
            dataList.append(newitem)
    return dataList


def getDistance(x, y, sigma = 1.0):
    # 高斯
    # d = np.sqrt(np.sum(np.square(x - y)))
    # w = np.exp(-(np.square(d) / np.square(sigma)))
    # return w

    # 欧氏距离
    d = np.sqrt(np.sum(np.square(x - y)))
    return d


def getDistanceMatrix(X, sigma = 1.0):
    X = np.array(X)
    nums = len(X)
    S = np.zeros((nums, nums))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            # S[i][j] = getDistance(X[i], X[j], sigma)  # 高斯
            S[i][j] = getDistance(X[i], X[j])  # 欧式距离
            S[j][i] = S[i][j]
    return S


def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id:
            d = S[i][j]
            w = np.exp(-(np.square(d) / np.square(sigma)))
            A[i][j] = w
            A[j][i] = w
        # for j in neighbours_id: # xj is xi's neighbour
        #     A[i][j] = np.exp(-S[i][j]/(sigma*sigma))
        #     A[j][i] = A[i][j] # mutually

    return A

def getAMatrix(M, k, sigma=1.0):
    nums = len(M)
    A = np.zeros([nums, nums])

    for i in range(nums):
        d = M[i][j]
        w = np.exp(-(np.square(d) / np.square(sigma)))
        A[i][j] = w
        A[j][i] = w

    for i in range(nums):
        l = M[i].tolist()
        neighbourL = l[:k+1]
        neighbourId = [0, 1, 2, 3, 4, 5]
        for x in range(k+1, len(l)):
            if l[x] < max(neighbourL):
                idx = neighbourL.index(max(neighbourL))
                neighbourL[idx] = l[x]
                neighbourId[idx] = x

        for j in neighbourId:
            d = M[i][j]
            w = np.exp(-(np.square(d) / np.square(sigma)))
            A[i][j] = w
            A[j][i] = w

        # A[i][i] = 0

    return A


def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # print degreeMatrix

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # print laplacianMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    # sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    # return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
    return laplacianMatrix


def getLPMatrix(M):
    nums = len(M)
    D = np.zeros([nums, nums])
    L = np.zeros([nums, nums])

    for i in range(nums):
        D[i][i] = sum(M[i])

    L = D - M
    return L


def draw(pcaData, clusters, ground = False):
    colorMark = ['red', 'blue', 'green', 'y', 'black', 'peru', 'blueviolet', 'fuchsia', 'olivedrab', 'silver',
                 'deepskyblue']
    if ground:
        plt.title('ground-truth')
    else:
        plt.title('spectral')

    # xl, yl = [], []
    # for key in clusters.keys():
    #     c = clusters[key]
    #     x, y = [], []
    #     for item in c:
    #         point = pcaData[int(item[0]) - 1]  # 根据id找到pca后的该点的坐标
    #         x.append(point[1])
    #         y.append(point[2])
    #     xl.append(x)
    #     yl.append(y)
    #
    # for i in range(len(clusters)):
    #     plt.scatter(xl[i], yl[i], s=50, label='class' + str(i + 1), c=colorMark[i], marker='.', alpha=None,
    #                 edgecolors='white')
    #
    # plt.legend()
    # plt.show()


    xl, yl = [], []
    for key in clusters.keys():
        c = clusters[key]
        x, y = [], []
        for item in c:
            point = pcaData[item]  # 根据id找到pca后的该点的坐标
            x.append(point[0])
            y.append(point[1])
        xl.append(x)
        yl.append(y)

    for i in range(len(clusters)):
        plt.scatter(xl[i], yl[i], s=50, label='class' + str(i + 1), c=colorMark[i], marker='.', alpha=None,
                    edgecolors='white')

    plt.legend()
    plt.show()


def compare(clusters, g_clusters, nums):
    m1, m2 = np.zeros([nums, nums]), np.zeros([nums, nums])
    list1, list2 = [], []

    for key in clusters.keys():
        l = []
        for i in clusters[key]:
            l.append(i)
        list1.append(l)

    for key in g_clusters.keys():
        l = []
        for i in g_clusters[key]:
            l.append(i)
        list2.append(l)

    for item in list1:
        x = 0
        while x < len(item):
            y = x
            while y < len(item):
                m1[item[x]][item[y]] = 1
                m1[item[y]][item[x]] = 1
                y += 1
            x += 1

    for item in list2:
        x = 0
        while x < len(item):
            y = x
            while y < len(item):
                m2[item[x]][item[y]] = 1
                m2[item[y]][item[x]] = 1
                y += 1
            x += 1

    m00, m01, m10, m11 = 0, 0, 0, 0
    for a in range(nums):
        for b in range(nums):
            if m1[a][b] == 1 and m2[a][b] == 1:
                m11 += 1
            elif m1[a][b] == 0 and m2[a][b] == 1:
                m01 += 1
            elif m1[a][b] == 1 and m2[a][b] == 0:
                m10 += 1
            else:
                m00 += 1

    r = (m11 + m00) / (m00 + m01 + m10 + m11)
    j = m11 / (m01 + m10 + m11)
    return r, j


if __name__ == '__main__':
    dataSet = loadDataSet('cho')
    k = 5
    data = np.array(dataSet)[:, 2:]
    label = np.array(dataSet)[:, 1]

    # 距离矩阵
    similarity = getDistanceMatrix(data)
    # print(similarity)
    # 邻接矩阵
    # Adjacent1 = myKNN(similarity, k=5)
    # print(Adjacent1)
    # print('------')
    Adjacent = getAMatrix(similarity, k)
    # print(Adjacent)
    # 拉普拉斯矩阵
    # Laplacian = calLaplacianMatrix(Adjacent)
    # print(Laplacian)
    # print('------')
    L = getLPMatrix(Adjacent)
    # print(L)

    x, V = np.linalg.eig(L)
    # print(x)
    # 将特征值排序
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0])
    # print(x)
    H = np.vstack([V[:, i] for (v, i) in x[:len(data)]]).T
    #
    sp_kmeans = KMeans(n_clusters=k).fit(H)
    # print(sp_kmeans.labels_)

    # print(data)
    clusters = {}
    for i, item in enumerate(sp_kmeans.labels_):
        if item not in clusters:
            clusters[item] = []
        clusters[item].append(i)


    print(clusters)

    g_clusters = {}
    for i, item in enumerate(dataSet):
        c = item[1]
        if c not in g_clusters.keys():
            g_clusters[c] = []
        g_clusters[c].append(i)
    print('---------')
    print(g_clusters)


    pca = PCA(n_components=2)
    pcaData = pca.fit_transform(data)
    pcaData = np.array(pcaData).tolist()
    # for i, item in enumerate(pcaData):
    #     item.insert(0, i + 1)



    # draw(pcaData, g_clusters, True)
    # draw(pcaData, clusters)

    # pure_kmeans = KMeans(n_clusters=5).fit(data)
    #
    # plot(data, sp_kmeans.labels_)

    r, j = compare(clusters, g_clusters, len(dataSet))
    print('randIndex is:' + str(r))
    print('jc is:' + str(j))